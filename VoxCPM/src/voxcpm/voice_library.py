"""离线音色库（Voice Library）：A 模式（reference-only）持久化与管理。

设计目标
--------
- 不修改 ``core.py`` / ``voxcpm.py`` / ``voxcpm2.py`` / ``audio_vae*.py``
- 只依赖 ``VoxCPM`` 实例暴露的 ``tts_model.build_prompt_cache()`` 与其
  ``_generate_with_prompt_cache()``
- 与 CLI 和 Gradio 服务共享同一 ``voices/`` 目录和 manifest，通过文件 mtime
  实现热加载
- 线程安全 + 进程安全（fcntl 文件锁）

存储结构
--------
::

    voices/
    ├── manifest.json          # 音色元数据（唯一索引）
    ├── .lock                  # fcntl 文件锁
    ├── alice.safetensors      # ref_audio_feat 张量
    └── bob.safetensors

只保存 AudioVAE 编码得到的 ``ref_audio_feat``，不保存原始音频，
也不保存 ``prompt_text``（A 模式不需要）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from safetensors.torch import load_file, save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class VoiceEntry:
    """单条音色的元数据（保存在 manifest.json 中）。"""

    id: str
    display_name: str
    source_file: str
    duration_sec: float
    sample_rate: int
    feat_file: str
    feat_shape: List[int]
    created_at: str
    model_revision: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceEntry":
        return cls(
            id=d["id"],
            display_name=d.get("display_name", d["id"]),
            source_file=d.get("source_file", ""),
            duration_sec=float(d.get("duration_sec", 0.0)),
            sample_rate=int(d.get("sample_rate", 16_000)),
            feat_file=d["feat_file"],
            feat_shape=list(d.get("feat_shape", [])),
            created_at=d.get("created_at", ""),
            model_revision=d.get("model_revision", ""),
        )


# --------------------------------------------------------------------------- #
# 文件锁（fcntl 优先，缺失则退化为无锁）
# --------------------------------------------------------------------------- #
class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        try:
            import fcntl

            self._fh = open(self.path, "a+")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------- #
# 核心类
# --------------------------------------------------------------------------- #
class VoiceLibrary:
    """离线音色库。线程安全（GIL + 文件锁），支持 CLI / UI 并发读写。"""

    def __init__(
        self,
        root_dir: str | Path,
        model=None,  # 可以是 None（只读模式），也可以是 VoxCPM 实例
        model_revision: str = "",
        auto_reload: bool = True,
    ):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".lock"

        self.model = model
        self.model_revision = model_revision
        self.auto_reload = auto_reload

        self._entries: Dict[str, VoiceEntry] = {}
        self._feats: Dict[str, torch.Tensor] = {}
        self._manifest_mtime: float = 0.0

        self._load_all()

    # ------------------------------------------------------------------ #
    # 构造：只读实例（不需要模型）
    # ------------------------------------------------------------------ #
    @classmethod
    def read_only(cls, root_dir: str | Path, model_revision: str = "") -> "VoiceLibrary":
        obj = cls.__new__(cls)
        obj.root = Path(root_dir)
        obj.root.mkdir(parents=True, exist_ok=True)
        obj.manifest_path = obj.root / "manifest.json"
        obj.lock_path = obj.root / ".lock"
        obj.model = None
        obj.model_revision = model_revision
        obj.auto_reload = False
        obj._entries = {}
        obj._feats = {}
        obj._manifest_mtime = 0.0
        obj._load_all()
        return obj

    def attach_model(self, model, model_revision: str = "") -> None:
        """模型加载后注入，切换到完整模式。"""
        self.model = model
        if model_revision:
            self.model_revision = model_revision
        # 重新加载 feats
        self._load_all()

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def register(
        self,
        voice_id: str,
        audio_path: str,
        display_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> VoiceEntry:
        """注册一个新音色。编码音频并落盘。"""
        if self.model is None:
            raise RuntimeError("VoiceLibrary 处于只读模式，无法注册音色")

        if not _ID_RE.match(voice_id):
            raise ValueError(
                f"voice_id 只能包含字母/数字/下划线/连字符，长度 1-64，实际: {voice_id!r}"
            )
        audio_path = str(audio_path)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"参考音频不存在: {audio_path}")

        with _FileLock(self.lock_path):
            self._reload_if_changed()
            if voice_id in self._entries and not overwrite:
                raise ValueError(f"voice_id 已存在: {voice_id}（用 overwrite=True 覆盖）")

            # 1. AudioVAE 编码
            cache = self.model.tts_model.build_prompt_cache(
                reference_wav_path=audio_path,
            )
            ref_feat = cache["ref_audio_feat"]
            if ref_feat.ndim != 3:
                raise RuntimeError(f"意外的 ref_audio_feat shape: {tuple(ref_feat.shape)}")
            ref_feat = ref_feat.detach().cpu().contiguous()

            # 2. 落盘张量
            feat_file = f"{voice_id}.safetensors"
            feat_path = self.root / feat_file
            if SAFETENSORS_AVAILABLE:
                save_file({"ref_audio_feat": ref_feat}, str(feat_path))
            else:
                torch.save({"ref_audio_feat": ref_feat}, str(feat_path))

            # 3. 读取时长 / 采样率（失败时估计）
            try:
                import soundfile as sf

                info = sf.info(audio_path)
                duration_sec = float(info.duration)
                sample_rate = int(info.samplerate)
            except Exception:
                duration_sec = float(ref_feat.shape[0]) * 0.16  # 粗略估计
                sample_rate = 16_000

            # 4. 更新 manifest
            entry = VoiceEntry(
                id=voice_id,
                display_name=display_name or voice_id,
                source_file=os.path.basename(audio_path),
                duration_sec=round(duration_sec, 3),
                sample_rate=sample_rate,
                feat_file=feat_file,
                feat_shape=list(ref_feat.shape),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                model_revision=self.model_revision,
            )
            self._entries[voice_id] = entry
            self._feats[voice_id] = ref_feat
            self._save_manifest()
            self._manifest_mtime = self.manifest_path.stat().st_mtime
            logger.info(f"[VoiceLibrary] registered: {voice_id} ({duration_sec:.2f}s)")
            return entry

    def remove(self, voice_id: str) -> bool:
        with _FileLock(self.lock_path):
            self._reload_if_changed()
            entry = self._entries.pop(voice_id, None)
            if entry is None:
                return False
            self._feats.pop(voice_id, None)
            feat_path = self.root / entry.feat_file
            if feat_path.exists():
                try:
                    feat_path.unlink()
                except OSError as e:
                    logger.warning(f"[VoiceLibrary] 删除 {feat_path} 失败: {e}")
            self._save_manifest()
            self._manifest_mtime = self.manifest_path.stat().st_mtime
            logger.info(f"[VoiceLibrary] removed: {voice_id}")
            return True

    def get(self, voice_id: str) -> dict:
        """返回 A 模式的 prompt_cache（dict），可直接传给 _generate_with_prompt_cache。"""
        if self.auto_reload:
            self._reload_if_changed()
        if voice_id not in self._entries:
            raise KeyError(f"音色不存在: {voice_id}")
        if voice_id not in self._feats:
            raise RuntimeError(
                f"音色 '{voice_id}' 的特征未加载（可能 manifest 与特征文件不一致）"
            )
        return {
            "ref_audio_feat": self._feats[voice_id],
            "mode": "reference",
        }

    def list(self) -> List[VoiceEntry]:
        if self.auto_reload:
            self._reload_if_changed()
        return list(self._entries.values())

    def has(self, voice_id: str) -> bool:
        if self.auto_reload:
            self._reload_if_changed()
        return voice_id in self._entries

    def ids(self) -> List[str]:
        return [e.id for e in self.list()]

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _load_all(self):
        self._entries.clear()
        self._feats.clear()

        if not self.manifest_path.exists():
            self._manifest_mtime = 0.0
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"[VoiceLibrary] manifest 读取失败: {e}")
            return

        manifest_rev = data.get("model_revision", "")
        if manifest_rev and self.model_revision and manifest_rev != self.model_revision:
            logger.warning(
                f"[VoiceLibrary] manifest 由 {manifest_rev} 构建，"
                f"当前模型为 {self.model_revision}，特征可能不兼容"
            )

        for vid, ed in data.get("voices", {}).items():
            try:
                entry = VoiceEntry.from_dict(ed)
                self._entries[vid] = entry
                if self.model is not None:
                    self._load_feat(vid, entry)
            except Exception as e:
                logger.error(f"[VoiceLibrary] 加载 {vid} 失败: {e}")

        self._manifest_mtime = self.manifest_path.stat().st_mtime
        logger.info(
            f"[VoiceLibrary] loaded {len(self._entries)} voices from {self.root}"
        )

    def _load_feat(self, vid: str, entry: VoiceEntry) -> None:
        feat_path = self.root / entry.feat_file
        if not feat_path.exists():
            logger.warning(f"[VoiceLibrary] 跳过 {vid}: 特征文件缺失 {feat_path}")
            return
        if SAFETENSORS_AVAILABLE and feat_path.suffix == ".safetensors":
            tensors = load_file(str(feat_path), device="cpu")
            self._feats[vid] = tensors["ref_audio_feat"]
        else:
            ckpt = torch.load(str(feat_path), map_location="cpu", weights_only=True)
            self._feats[vid] = ckpt["ref_audio_feat"]

    def _save_manifest(self) -> None:
        data = {
            "version": MANIFEST_VERSION,
            "model_revision": self.model_revision,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "voices": {vid: e.to_dict() for vid, e in self._entries.items()},
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.manifest_path)

    def _reload_if_changed(self) -> None:
        if not self.manifest_path.exists():
            if self._entries:
                self._load_all()
            return
        mtime = self.manifest_path.stat().st_mtime
        if mtime != self._manifest_mtime:
            logger.info("[VoiceLibrary] manifest 变化，热加载 …")
            self._load_all()

