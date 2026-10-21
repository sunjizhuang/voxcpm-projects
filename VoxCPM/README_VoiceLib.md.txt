
一、总体设计
┌─────────────────────────────────────────────────────────────┐
│                     离线阶段（一次性）                        │
│  CLI:  voxcpm voice register --id alice --audio ref.wav      │
│  或 UI: 上传音频 + 填 id + 点“注册”                           │
│         ↓                                                    │
│  tts_model.build_prompt_cache(reference_wav_path=...)        │
│         ↓  {"ref_audio_feat": (T,P,D), "mode": "reference"}  │
│  写盘: voices/alice.safetensors + voices/manifest.json       │
└─────────────────────────────────────────────────────────────┘

                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     在线阶段（每次请求）                      │
│  客户端: voice_id="alice" + target_text + (control)          │
│         ↓                                                    │
│  VoiceLibrary.get("alice") → 命中内存缓存（或 mtime 变化后  │
│  自动 reload）                                               │
│         ↓                                                    │
│  tts_model._generate_with_prompt_cache(                     │
│      target_text=final_text, prompt_cache=cache, ...)       │
│         ↓                                                    │
│  decode_audio → numpy → Gradio Audio                         │
└─────────────────────────────────────────────────────────────┘

二、存储格式




# VoxCPM 音色库（Voice Library）功能说明

> 在 VoxCPM2 已有三种语音生成方式（Voice Design / Controllable Cloning /
> Ultimate Cloning）的基础上，新增 **第四种方式：音色库（Voice Library）**。
> 客户端只需传入 `voice_id` + `target_text`，即可使用服务端预注册的固定音色合成，
> 无需每次上传参考音频，也无需维护转写文本。

---

## 目录

- [1. 功能概览](#1-功能概览)
- [2. 设计原则](#2-设计原则)
- [3. 安装与依赖](#3-安装与依赖)
- [4. 目录结构与 manifest](#4-目录结构与-manifest)
- [5. CLI 用法](#5-cli-用法)
- [6. Web UI 用法](#6-web-ui-用法)
- [7. 编程接口](#7-编程接口)
- [8. 热加载与并发](#8-热加载与并发)
- [9. 常见问题](#9-常见问题)
- [10. 限制与注意事项](#10-限制与注意事项)

---

## 1. 功能概览

| 模式 | 输入 | 是否上传音频 | 是否需要转写 | 典型用途 |
|------|------|:------------:|:------------:|----------|
| 🎨 Voice Design | Control Instruction + Target Text | 否 | 否 | 从零创造新声音 |
| 🎛️ Controllable Cloning | Reference Audio + (可选) Control + Text | 是 | 否 | 保留音色 + 控制情感/
语速 |

| 🎙️ Ultimate Cloning | Reference Audio + 转写 + Text | 是 | **是** | 
高保真还原韵律、情感、细节 |

| **🗂️ Voice Library（新增）** | **voice_id + (可选) Control + Text** | **否** | **否** | **
多音色复用、免上传、免维护转写** |


音色库内部使用 VoxCPM2 的 **A 模式（reference-only）** 缓存：

- 参考音频通过 `VoxCPM2Model.build_prompt_cache(reference_wav_path=...)` 编码为
  `ref_audio_feat`，通过 103/104 token 结构隔离注入，**不需要转写文本**
- 生成时直接调用 `VoxCPM2Model._generate_with_prompt_cache()`，**不修改核心推理代码**

---

## 2. 设计原则

1. **最小侵入**：仅新增 `voxcpm/voice_library.py`，修改 `cli.py` 与 `app.py`。
   `core.py` / `voxcpm.py` / `voxcpm2.py` / `audio_vae*.py` 完全不动。
2. **CLI 与 UI 共享**：两者读写同一个 `voices/` 目录与 `manifest.json`，
   通过文件 mtime 检测实现热加载。
3. **热加载**：UI 注册后立即可用；CLI 注册后下次请求即生效，无需重启服务。
4. **线程安全 + 进程安全**：Python GIL + `fcntl.flock` 文件锁。
5. **持久化特征**：只保存 AudioVAE 编码后的 `ref_audio_feat`，不保存原始音频，
   服务启动时直接加载张量，跳过 AudioVAE 编码。

---

## 3. 安装与依赖

新增依赖 **无**。已有依赖即可：

- `torch`
- `safetensors`（推荐；缺失时自动退化为 `torch.save` / `torch.load`）
- `soundfile`（用于读取参考音频时长，缺失时用估算值）
- `gradio`（仅 UI 需要）

确保 `voxcpm/voice_library.py` 放在正确位置：


---

## 4. 目录结构与 manifest

### 4.1 目录结构

默认目录为 `./voices/`，可通过环境变量 `VOXCPM_VOICES_DIR` 或 CLI 的
`--voices-dir` 覆盖。
voices/
├── manifest.json          # 音色元数据（唯一索引）
├── .lock                  # fcntl 文件锁（跨进程写保护）
├── alice.safetensors      # ref_audio_feat 张量
├── bob.safetensors
└── ...


### 4.2 `manifest.json` 格式

```json
{
  "version": 1,
  "model_revision": "openbmb/VoxCPM2",
  "updated_at": "2026-10-08T12:34:56Z",
  "voices": {
    "alice": {
      "id": "alice",
      "display_name": "Alice",
      "source_file": "alice_ref.wav",
      "duration_sec": 5.21,
      "sample_rate": 16000,
      "feat_file": "alice.safetensors",
      "feat_shape": [130, 4, 64],
      "created_at": "2026-10-08T12:34:56Z",
      "model_revision": "openbmb/VoxCPM2"
    }
  }
}

字段	说明
id	音色 ID（唯一键，字母/数字/_/-，长度 1–64）
display_name	人类可读名称
source_file	注册时使用的参考音频文件名（仅记录，不落盘）
duration_sec	参考音频时长（秒）
sample_rate	参考音频原始采样率
feat_file	张量文件名
feat_shape	ref_audio_feat 的 shape (T, P, D)
created_at	ISO 8601 时间戳（UTC）
model_revision	构建时的模型标识，用于版本校验

张量文件
T：patch 数（约等于音频时长 / 0.16 秒）
P：patch_size（VoxCPM2 为 4）
D：AudioVAE latent dim（VoxCPM2 为 64）
10 秒参考音频约 40 KB，1000 个音色约 40 MB。


5. CLI 用法

5.1 注册音色
voxcpm voice register --id alice --audio ref.wav [--display-name "Alice"] [--voices-dir ./voices]

可选参数（与 clone / design 共享）：
--model-path / --hf-model-id / --device / --cache-dir /
--local-files-only / --no-denoiser / --no-optimize / --zipenhancer-path
--voices-dir（默认 ./voices）
    
5.2 列出音色
voxcpm voice list [--voices-dir ./voices]

ID                       Display                  Duration Created
--------------------------------------------------------------------------------
alice                    Alice                      5.21s 2026-10-08T12:34:56Z
bob                      Bob                        4.87s 2026-10-08T12:35:12Z

5.3 查看详情
voxcpm voice info --id alice [--voices-dir ./voices]

5.4 删除音色
voxcpm voice remove --id alice [--voices-dir ./voices]

删除 manifest 条目 + 张量文件。


6. Web UI 用法
python app.py --host 0.0.0.0 --port 8808 --voices-dir ./voices

6.1 使用音色库生成
在 🗂️ 音色库 下拉框中选择一个已注册音色（例如 alice）

上传音频区域会自动禁用（互斥）

填写 Target Text（可选用 Control Instruction 控制情感/语速）

点击 🔊 开始生成

6.2 在线注册新音色
展开 🗂️ 音色库管理 面板：

填写 Voice ID（如 alice）

可选填写 Display Name

上传 参考音频（wav / mp3 / m4a 均可）

点击 ✅ 注册音色

注册完成后：

状态栏显示 ✅ 已注册 alice

🗂️ 音色库 下拉框自动刷新，立即可选用（热加载）

6.3 删除音色
在 🗂️ 音色库管理 面板：

从 选择要删除的音色 下拉框中选择目标

点击 🗑️ 删除

6.4 刷新列表
如果通过 CLI 在外部注册了音色，点击 🔄 刷新列表 即可看到。

