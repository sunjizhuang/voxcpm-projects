from voxcpm import VoxCPM
import soundfile as sf
import numpy as np
import sounddevice as sd
from datetime import datetime
import time


def print_time(message):
    """打印当前准确时间（精确到毫秒）"""
    now = datetime.now()
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {message}", flush=True)

PROMPT_TEXT = "今天天气很好"

print_time("Loading VoxCPM2...")
load_start = time.perf_counter()

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda",
    #device="cpu",
    load_denoiser=False,
    optimize=True,         #启用 torch.compile 来加速模型推理， 它会对 VoxCPM 的模型计算图进行编译优化，从而提升运行速度，CUDA（GPU）环境有意义
)

print_time(
    f"VoxCPM2 loaded, 耗时 {time.perf_counter() - load_start:.3f} 秒"
)

print_time("Generating...")
generate_start = time.perf_counter()

wav = model.generate(
   # text="(A young woman, gentle and sweet voice)你好，这是一段灵境具身科技公司生成的运行在 RTX 4080 上的 VoxCPM 语音。我们是一家致力于机器人大脑的探索与开发AI企业",
    text="(A young woman, gentle and sweet voice)今天天气很好",
    cfg_value=2.0,
    inference_timesteps=10,
)
print_time(
    f"Generation finished, 耗时 {time.perf_counter() - generate_start:.3f} 秒"
)
save_start = time.perf_counter()

sf.write(
    "female_voice.wav",
    wav,
    model.tts_model.sample_rate
)

print_time(
    f"Saved: test_tts.wav, 保存耗时 {time.perf_counter() - save_start:.3f} 秒"
)
#print("生成完成：test.wav")

###############Generate Sound 2##########
print_time("Generating...2")
generate_start = time.perf_counter()

wav = model.generate(
   # text="(A young man, gentle and sweet voice)你好，这是一段灵境具身科技公司生成的运行在 RTX 4080 上的 VoxCPM 语音。我们是一家致力于机器人大脑的探索与开发AI企业",
    text="今天天气很好，这是一段运行在 Ubuntu 22.04 和 RTX 4080 上的 VoxCPM2 流式语音。英格兰牛津郡的皮丁顿村近日举行了一次象征性的“公民投票”，宣布从英国“独立”，并更名为“皮丁顿公国”。这一决定源于英国政府计划在该村附近的一个旧军事设施安置1256名男性移民，引发该村居民的强烈不满。村民通过这次投票表达抗议，认为这一安排威胁到他们的安全、保障和自由",
    cfg_value=2.0,
    inference_timesteps=10,
)
print_time(
    f"Generation finished, 耗时 {time.perf_counter() - generate_start:.3f} 秒"
)
save_start = time.perf_counter()

sf.write(
    "test_TTS_2.wav",
    wav,
    model.tts_model.sample_rate
)

print_time(
    f"Saved: test_tts.wav, 保存耗时 {time.perf_counter() - save_start:.3f} 秒"
)
#############################################

# ============================================================
# 只执行一次
# ============================================================

print_time(f"Building prompt cache...")

prompt_cache = model.tts_model.build_prompt_cache(
    prompt_text=PROMPT_TEXT,
    prompt_wav_path="./female_voice.wav",
)
############### Streaming Sound Playback ##########

def generate_and_play_streaming(model, text):
    """使用 VoxCPM2 流式生成，并通过 Ubuntu 22.04 默认声卡实时播放。"""
    print_time("Streaming TTS + playback started...")
    stream_start = time.perf_counter()

    sample_rate = model.tts_model.sample_rate
    print_time(f"Audio sample rate: {sample_rate} Hz")

    # sounddevice 直接打开系统默认输出设备，不需要额外安装播放器。
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        latency="low",
    ) as stream:

        chunk_count = 0
        total_samples = 0

        for chunk in model.generate_streaming(
            text=text,
            cfg_value=2.0,
            inference_timesteps=10,
        ):

       # for chunk, _, _ in model.tts_model.generate_with_prompt_cache_streaming(
       #     target_text=text,
       #     prompt_cache=prompt_cache,
       #     cfg_value=2.0,
       #     inference_timesteps=4,
       # ):
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1, 1)

            if chunk.size == 0:
                continue

            stream.write(chunk)

            chunk_count += 1
            total_samples += chunk.shape[0]

            if chunk_count % 10 == 0:
                print_time(
                    f"Streaming chunk={chunk_count}, "
                    f"audio={total_samples / sample_rate:.2f}s"
                )

    print_time(
        f"Streaming playback finished, "
        f"耗时 {time.perf_counter() - stream_start:.3f} 秒, "
        f"音频时长 {total_samples / sample_rate:.2f} 秒"
    )


print_time("Generating and playing streaming voice...")

generate_and_play_streaming(
    model,
    "(A young woman, gentle and sweet voice) "
    "今天天气很好，这是一段运行在 Ubuntu 22.04 和 RTX 4080 上的 VoxCPM2 流式语音。英格兰牛津郡的皮丁顿村近日举行了一次象征性的“公民投票”，宣布从英国“独立”，并更名为“皮丁顿公国”。这一决定源于英国政府计划在该村附近的一个旧军事设施安置1256名男性移民，引发该村居民的强烈不满。村民通过这次投票表达抗议，认为这一安排威胁到他们的安全、保障和自由"
)

#############################################
