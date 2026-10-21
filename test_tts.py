from voxcpm import VoxCPM
import soundfile as sf
from datetime import datetime
import time


def print_time(message):
    """打印当前准d确时间（精确到毫秒）"""
    now = datetime.now()
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {message}", flush=True)

print_time("Loading VoxCPM2...")
load_start = time.perf_counter()

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda",
    #device="cpu",
    load_denoiser=False,
)

print_time(
    f"VoxCPM2 loaded, 耗时 {time.perf_counter() - load_start:.3f} 秒"
)

print_time("Generating...")
generate_start = time.perf_counter()

wav = model.generate(
   # text="(A young woman, gentle and sweet voice)你好，这是一段灵境具身科技公司生成的运行在 RTX 4080 上的 VoxCPM 语音。我们是一家致力于机器人大脑的探索与开发AI企业",
    text="(A young woman, gentle and sweet voice)A young woman, gentle and sweet voice , jin tian tian qi hen hao, 今天天气很好)",
    cfg_value=2.0,
    inference_timesteps=10,
)
print_time(
    f"Generation finished, 耗时 {time.perf_counter() - generate_start:.3f} 秒"
)
save_start = time.perf_counter()

sf.write(
    "test_TTS.wav",
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
    text="(A young woman, gentle and sweet voice) 今天天气很好)",
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
