import os
import re
import sys
import logging
import random
import numpy as np
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device, next_and_close
from voxcpm.voice_library import VoiceLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_VOICES_DIR = os.environ.get("VOXCPM_VOICES_DIR", "./voices")

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**AetherLin — Four Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and AetherLin "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail. "
    "Note: This mode will disable Control Instruction.\n\n"
    "🗂️ **Voice Library** — Reuse pre-registered voices by ID  \n"
    "Pick a voice from the **Voice Library** dropdown. No audio upload is required. "
    "Mutually exclusive with the reference audio uploader. "
    "Use the **Voice Library Management** panel to register new voices on the fly."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    "`Target Text`: *\"I never asked you to stay… It's not like I care or anything. "
    "But… why does it still hurt so much now that you're gone?\"*  \n\n"
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**AetherLin — 四种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），AetherLin 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用 Control Instruction。\n\n"
    "🗂️ **音色库（Voice Library）**  \n"
    "从 **音色库** 下拉框中选择一个已注册的音色即可，无需上传音频。"
    "音色库模式与上传参考音频互斥。"
    "可通过 **音色库管理** 面板在线注册新音色。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "seed_label": "Seed",
        "seed_info": "Seed used for reproducible generation. Updated with the actual successful seed after generation.",
        "random_seed_label": "Random Seed",
        "random_seed_info": "Generate a new seed before each inference run.",
        "voice_library_label": "🗂️ Voice Library (mutually exclusive with reference audio)",
        "voice_library_info": "Pick a pre-registered voice by ID. No audio upload needed.",
        "voice_none_option": "(do not use voice library)",
        "voice_manage_title": "🗂️ Voice Library Management",
        "voice_id_label": "Voice ID",
        "voice_id_placeholder": "alice",
        "voice_display_name_label": "Display Name (optional)",
        "voice_display_name_placeholder": "Alice",
        "voice_register_audio_label": "Reference Audio for Registration",
        "voice_register_btn": "✅ Register Voice",
        "voice_refresh_btn": "🔄 Refresh List",
        "voice_delete_label": "Select Voice to Delete",
        "voice_delete_btn": "🗑️ Delete",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "voice_library_label": "🗂️ 音色库（与上传参考音频互斥）",
        "voice_library_info": "从离线音色库中选择一个已注册音色，无需上传音频",
        "voice_none_option": "（不使用音色库）",
        "voice_manage_title": "🗂️ 音色库管理",
        "voice_id_label": "音色 ID",
        "voice_id_placeholder": "alice",
        "voice_display_name_label": "显示名称（可选）",
        "voice_display_name_placeholder": "Alice",
        "voice_register_audio_label": "用于注册的参考音频",
        "voice_register_btn": "✅ 注册音色",
        "voice_refresh_btn": "🔄 刷新列表",
        "voice_delete_label": "选择要删除的音色",
        "voice_delete_btn": "🗑️ 删除",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,
    "zh": None,
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "AetherLin TTS is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 100px;
    width: auto;
    max-width: 250px;
    display: inline-block;
}

.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------


class VoxCPMDemo:
    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        device: str = "auto",
        voices_dir: str = DEFAULT_VOICES_DIR,
    ) -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

        self._voices_dir = voices_dir
        # 只读模式先加载（用于 UI 渲染下拉框）；模型加载后再 attach_model
        self.voice_library: VoiceLibrary = VoiceLibrary.read_only(voices_dir)
        logger.info(
            f"Voice library loaded from {voices_dir} "
            f"({len(self.voice_library.ids())} voices)"
        )

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("Model loaded successfully.")
        # 将模型注入 voice library，切换到完整模式
        self.voice_library.attach_model(self.voxcpm_model, self._model_id)
        return self.voxcpm_model

    def get_or_load_asr_model(self):
        if self.asr_model is not None:
            return self.asr_model
        from funasr import AutoModel

        logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)

    # ------------------------------------------------------------------ #
    # 第四态：音色库生成
    # ------------------------------------------------------------------ #
    def generate_from_voice_library(
        self,
        text_input: str,
        voice_id: str,
        control_instruction: str = "",
        cfg_value_input: float = 2.0,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        """使用离线音色库（A 模式 reference-only）生成语音。

        直接调用 tts_model._generate_with_prompt_cache，绕过 core.VoxCPM._generate，
        实现"最小侵入、不动核心推理代码"。
        """
        if self.voice_library is None:
            raise RuntimeError("VoiceLibrary 未初始化")

        # 确保模型已加载（同时把模型注入 voice library）
        current_model = self.get_or_load_voxcpm()

        cache = self.voice_library.get(voice_id)

        text = (text_input or "").strip()
        if not text:
            raise ValueError("Please input text to synthesize.")

        control = re.sub(r"[()（）]", "", (control_instruction or "").strip()).strip()
        final_text = f"({control}){text}" if control else text

        logger.info(
            f"[Voice Library] voice_id='{voice_id}' "
            f"control='{control[:40] if control else 'None'}'"
        )
        logger.info(f"Generating audio for text: '{final_text[:80]}...'")

        gen = current_model.tts_model._generate_with_prompt_cache(
            target_text=final_text,
            prompt_cache=cache,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps),
            seed=seed,
        )
        try:
            decode_audio, _, _ = next_and_close(gen)
        except TypeError:
            # 兼容老版本 next_and_close 不存在的情况
            decode_audio, _, _ = next(gen)
            gen.close()

        wav_np = decode_audio.squeeze(0).cpu().numpy()
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return current_model.tts_model.sample_rate, wav_np, last_successful_seed


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        voice_id: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
    ):
        seed = _coerce_seed(seed_value)

        # ---- 第四态：音色库 ----
        if voice_id and not ref_wav:
            sr, wav_np, last_successful_seed = demo.generate_from_voice_library(
                text_input=text,
                voice_id=voice_id,
                control_instruction=control_instruction,
                cfg_value_input=cfg_value,
                inference_timesteps=int(dit_steps),
                seed=seed,
            )
            return (sr, wav_np), last_successful_seed

        # ---- 前三态 ----
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        sr, wav_np, last_successful_seed = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            seed=seed,
        )
        return (sr, wav_np), last_successful_seed

    def _on_toggle_instant(checked):
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    # ---------------- Voice library handlers ----------------
    def _on_voice_id_change(vid):
        """音色库选中 → 禁用 ref_wav / show_prompt_text；反之恢复。"""
        if vid:
            return (
                gr.update(value=None, interactive=False),   # reference_wav
                gr.update(value=False, interactive=False),  # show_prompt_text
            )
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def _on_ref_wav_change(ref_wav):
        """上传了音频 → 清空 voice_id 下拉框。"""
        if ref_wav:
            return gr.update(value="")
        return gr.update()

    def _register_voice(vid, name, audio):
        if not vid or not vid.strip():
            return "❌ 请填写 Voice ID", gr.update(), gr.update()
        if not audio:
            return "❌ 请上传参考音频", gr.update(), gr.update()
        try:
            # ← 新增：确保模型已加载，并把模型注入 voice_library
            demo.get_or_load_voxcpm()		
            demo.voice_library.register(
                voice_id=vid.strip(),
                audio_path=audio,
                display_name=(name or "").strip() or None,
                overwrite=False,
            )
            new_choices = [""] + demo.voice_library.ids()
            delete_choices = demo.voice_library.ids()
            return (
                f"✅ 已注册 `{vid.strip()}`",
                gr.update(choices=new_choices),
                gr.update(choices=delete_choices),
            )
        except Exception as e:
            logger.exception("Register voice failed")
            return f"❌ 注册失败：{e}", gr.update(), gr.update()

    def _delete_voice(vid):
        if not vid:
            return "❌ 未选择音色", gr.update(), gr.update(), gr.update()
        try:
            demo.voice_library.remove(vid)
            new_choices = [""] + demo.voice_library.ids()
            delete_choices = demo.voice_library.ids()
            return (
                f"🗑️ 已删除 `{vid}`",
                gr.update(choices=new_choices, value=""),
                gr.update(choices=delete_choices),
                gr.update(choices=delete_choices, value=None),
            )
        except Exception as e:
            logger.exception("Delete voice failed")
            return f"❌ 删除失败：{e}", gr.update(), gr.update(), gr.update()

    def _refresh_voices():
        ids = demo.voice_library.ids()
        return gr.update(choices=[""] + ids), gr.update(choices=ids)

    # ---------------- Build UI ----------------
    initial_ids = demo.voice_library.ids()

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )

                voice_id_dropdown = gr.Dropdown(
                    choices=[""] + initial_ids,
                    value="",
                    label=I18N("voice_library_label"),
                    info=I18N("voice_library_info"),
                    allow_custom_value=False,
                )

                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )
                    with gr.Row():
                        seed_value = gr.Number(
                            value=random.randint(0, 2**32 - 1),
                            precision=0,
                            label=I18N("seed_label"),
                            info=I18N("seed_info"),
                            interactive=False,
                        )
                        random_seed = gr.Checkbox(
                            value=True,
                            label=I18N("random_seed_label"),
                            elem_classes=["switch-toggle"],
                            info=I18N("random_seed_info"),
                        )

                with gr.Accordion(I18N("voice_manage_title"), open=False):
                    with gr.Row():
                        new_voice_id = gr.Textbox(
                            label=I18N("voice_id_label"),
                            placeholder=I18N("voice_id_placeholder"),
                            scale=1,
                        )
                        new_voice_name = gr.Textbox(
                            label=I18N("voice_display_name_label"),
                            placeholder=I18N("voice_display_name_placeholder"),
                            scale=1,
                        )
                    new_voice_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label=I18N("voice_register_audio_label"),
                    )
                    with gr.Row():
                        register_btn = gr.Button(I18N("voice_register_btn"), variant="secondary")
                        refresh_btn = gr.Button(I18N("voice_refresh_btn"), variant="secondary")
                    with gr.Row():
                        delete_voice_id = gr.Dropdown(
                            choices=initial_ids,
                            value=None,
                            label=I18N("voice_delete_label"),
                            scale=2,
                        )
                        delete_btn = gr.Button(I18N("voice_delete_btn"), variant="stop", scale=1)
                    manage_status = gr.Markdown("")

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        # ---------------- Event bindings ----------------
        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        random_seed.change(
            fn=_on_random_seed_toggle,
            inputs=[random_seed],
            outputs=[seed_value],
        )

        # 音色库 ↔ 上传音频 互斥
        voice_id_dropdown.change(
            fn=_on_voice_id_change,
            inputs=[voice_id_dropdown],
            outputs=[reference_wav, show_prompt_text],
        )
        reference_wav.change(
            fn=_on_ref_wav_change,
            inputs=[reference_wav],
            outputs=[voice_id_dropdown],
        )

        # 音色库管理
        register_btn.click(
            fn=_register_voice,
            inputs=[new_voice_id, new_voice_name, new_voice_audio],
            outputs=[manage_status, voice_id_dropdown, delete_voice_id],
        )
        refresh_btn.click(
            fn=_refresh_voices,
            inputs=[],
            outputs=[voice_id_dropdown, delete_voice_id],
        )
        delete_btn.click(
            fn=_delete_voice,
            inputs=[delete_voice_id],
            outputs=[manage_status, voice_id_dropdown, delete_voice_id, delete_voice_id],
        )

        run_btn.click(
            fn=_prepare_seed,
            inputs=[random_seed, seed_value],
            outputs=[seed_value],
            show_progress=False,
        ).then(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                voice_id_dropdown,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                seed_value,
            ],
            outputs=[audio_output, seed_value],
            show_progress=True,
            api_name="generate",
        )

    return interface


def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
    voices_dir: str = DEFAULT_VOICES_DIR,
):
    demo = VoxCPMDemo(model_id=model_id, device=device, voices_dir=voices_dir)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address. Use 127.0.0.1 to restrict access to the local machine; "
             "the default 0.0.0.0 exposes the unauthenticated UI/API to the network (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--voices-dir",
        type=str,
        default=DEFAULT_VOICES_DIR,
        help=f"Voice library directory (default: {DEFAULT_VOICES_DIR})",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
        voices_dir=args.voices_dir,
    )
