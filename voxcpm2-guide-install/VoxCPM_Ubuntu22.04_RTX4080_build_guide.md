# Ubuntu 22.04 + RTX 4080 部署 VoxCPM2 指导文档

Ubuntu 22.04 + RTX 4080 部署 VoxCPM 是比较合适的组合，RTX 4080 属于 Ada Lovelace，Compute Capability 8.9，VoxCPM 的 CUDA 推理可以直接利用它。
如果你的目标是中文 TTS + 声音克隆 + 后续提供 HTTP/WebSocket 服务给机器人使用，我建议直接按 VoxCPM2 + CUDA + Python 虚拟环境来部署，而不是先折腾 C++ 版本。

## 1. 目标

本文用于指导在 **Ubuntu 22.04 + NVIDIA RTX 4080** 环境部署 VoxCPM，重点面向：

- 中文 TTS（文本转语音）
- 声音克隆
- GPU 推理
- Streaming 实时语音合成
- 后续提供 HTTP / WebSocket TTS 服务
- 后续与机器人控制系统（例如 Lite3）集成

推荐方案：

```text
Ubuntu 22.04
    ↓
NVIDIA Driver
    ↓
CUDA / PyTorch CUDA
    ↓
Python 虚拟环境
    ↓
VoxCPM2
    ↓
GPU 推理
    ↓
TTS / 声音克隆 / Streaming
    ↓
HTTP / WebSocket TTS Server
    ↓
机器人
```

---

## 2. 推荐软件环境

建议优先采用：

| 组件 | 推荐 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA RTX 4080 |
| GPU 架构 | Ada Lovelace |
| Compute Capability | 8.9 |
| Python | 3.10 / 3.11 |
| PyTorch | >= 2.5 |
| CUDA | >= 12.0 |
| VoxCPM | VoxCPM2 |
| Python 环境 | venv |
| 推理设备 | CUDA GPU |

RTX 4080 对 CUDA 8.9 架构支持良好，一般不需要针对 RTX 4080 做特殊适配。

> 注意：`nvidia-smi` 显示的 CUDA Version 与本机 `nvcc --version` 显示的 CUDA Toolkit 版本不是一回事。最终应重点确认 PyTorch 是否能够正常使用 CUDA。

---

# 3. 部署前检查

在 RTK4080 上执行以下命令。

## 3.1 检查 Ubuntu

```bash
cat /etc/os-release
```

确认类似：

```text
Ubuntu 22.04
```

也可以：

```bash
lsb_release -a
```

---

## 3.2 检查 CPU 架构

```bash
uname -m
```

常见结果：

```text
x86_64
```

RTX 4080 PC 通常是 x86_64。

---

## 3.3 检查 NVIDIA Driver

```bash
nvidia-smi
```

重点确认：

```text
GPU Name: NVIDIA GeForce RTX 4080
Driver Version: xxx
CUDA Version: xxx
```

如果 `nvidia-smi` 无法正常执行，应先解决 NVIDIA Driver 问题，再继续 VoxCPM 安装。

---

## 3.4 检查 CUDA Toolkit

```bash
nvcc --version
```

注意：

```text
nvidia-smi
```

和：

```text
nvcc --version
```

显示的 CUDA 版本可能不同，这是正常的。

---

## 3.5 检查 Python

```bash
python3 --version
```

建议使用：

```text
Python 3.10.x
```

或者：

```text
Python 3.11.x
```

VoxCPM 当前推荐 Python 3.10–3.12。

---

# 4. 建议建立独立 Python 环境

不要直接修改 Ubuntu 系统 Python。

创建 VoxCPM 工作目录：

```bash
mkdir -p ~/voxcpm
cd ~/voxcpm
```

创建虚拟环境：

```bash
python3 -m venv .venv
```

激活：

```bash
source .venv/bin/activate
```

检查：

```bash
which python
```

应该类似：

```text
/home/<user>/voxcpm/.venv/bin/python
```

升级基础工具：

```bash
pip install -U pip setuptools wheel
```

以后每次进入 VoxCPM 环境：

```bash
cd ~/voxcpm
source .venv/bin/activate
```

退出：

```bash
deactivate
```

---

# 5. 安装 PyTorch

这是部署过程中最关键的一步。

**先安装并验证 PyTorch CUDA，不要急着安装 VoxCPM。**

根据 NVIDIA Driver 情况选择合适的 PyTorch CUDA wheel。

例如 CUDA 12.6：

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
```
安装可能需要一些时间。




如果当前环境适合 CUDA 13：

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
```

也可以根据 PyTorch 官方安装页面选择当前推荐版本。

---

# 6. 验证 PyTorch + RTX 4080

执行：

完成后不要安装 VoxCPM，先验证 PyTorch。

执行：
```bash
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
```

我们希望看到类似：
torch: 2.x.x
torch CUDA: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 4080
Capability: (8, 9)

```

其中最重要的是：

```text
cuda available: True
GPU: NVIDIA GeForce RTX 4080
capability: (8, 9)
```

如果：

```text
cuda available: False
```

**不要继续安装 VoxCPM。**

此时先排查：

1. NVIDIA Driver
2. PyTorch 安装版本
3. CUDA wheel
4. GPU 是否被系统识别

---

# 7. 安装 VoxCPM

PyTorch CUDA 验证成功以后：

```bash
pip install voxcpm
```

安装完成后验证：

```bash
python -c "from voxcpm import VoxCPM; print('VoxCPM OK')"
```

正常：

```text
VoxCPM OK
```

---

# 8. 下载 VoxCPM2 模型

VoxCPM2 是当前优先推荐使用的模型。

可以通过 Python 自动下载：

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda"
)
```
或者
```bash
python -c "from voxcpm import VoxCPM; model = VoxCPM.from_pretrained('openbmb/VoxCPM2', device='cuda')"
```

第一次运行时会下载模型。

模型通常会进入 Hugging Face 本地缓存目录。

---

# 9. Hugging Face 下载较慢时

如果服务器访问 Hugging Face 不稳定，可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

也可以写入：

```bash
~/.bashrc
```

例如：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

然后再运行模型下载程序。

如果所在网络能够正常访问 Hugging Face，则不需要配置镜像。

或者
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer
用 hf_transfer 有时能明显提速。

---

# 10. 第一次 TTS 测试

创建测试程序：

```bash
cd ~/voxcpm
nano test_tts.py
```

写入：

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda",
    load_denoiser=False,
)

wav = model.generate(
    text="你好，这是一段运行在 RTX 4080 上的 VoxCPM 语音。",
    cfg_value=2.0,
    inference_timesteps=10,
)

sf.write(
    "test.wav",
    wav,
    model.tts_model.sample_rate
)

print("生成完成：test.wav")
```

运行：

```bash
python test_tts.py
```

正常：

```text
生成完成：test.wav
```

确认文件：

```bash
ls -lh test.wav
```

如果系统有 ALSA 播放设备：

```bash
aplay test.wav
```

---

# 11. 验证模型是否使用 RTX 4080

另开一个终端：

```bash
watch -n 0.5 nvidia-smi
```

另一个终端执行：

```bash
cd ~/voxcpm
source .venv/bin/activate
python test_tts.py
```

观察 `nvidia-smi`：

```text
GPU: NVIDIA GeForce RTX 4080
Memory-Usage: xxxxMiB / xxxxxMiB
```

如果模型加载后显存明显增加，说明模型已经进入 GPU。

---

# 12. 声音克隆测试

VoxCPM 的重要功能之一是基于参考音频进行声音克隆。

准备参考音频：

```text
reference.wav
```

建议：

- 单人声音
- 尽量没有背景音乐
- 尽量没有环境噪声
- 普通说话
- 语音清晰
- 建议准备约 10～20 秒参考音频

逻辑：

```text
参考音频
    +
参考音频对应文字
    +
目标文字
    ↓
VoxCPM2
    ↓
目标语音
```

后续可以根据当前安装的 VoxCPM2 API 测试 voice cloning。

---

# 13. Streaming 实时语音

如果最终用于机器人，不建议一直采用：

```text
文本
 ↓
生成完整 WAV
 ↓
保存文件
 ↓
播放
```

更适合机器人的是：

```text
文本
 ↓
VoxCPM Streaming
 ↓
PCM chunk
 ↓
Audio Buffer
 ↓
ALSA / 音频设备
 ↓
扬声器
```

这样可以降低首包语音延迟。

后续部署 TTS Server 时建议优先考虑 Streaming API。

---

# 14. 后续机器人集成架构

如果最终用于 Lite3 或其他机器人，建议 RTX4080 作为独立语音服务器：

```text
                    RTX4080 PC
┌──────────────────────────────────────┐
│ Ubuntu 22.04                         │
│                                      │
│ NVIDIA Driver                        │
│       ↓                              │
│ PyTorch CUDA                         │
│       ↓                              │
│ VoxCPM2                              │
│       ↓                              │
│ TTS Server                           │
│       ↓                              │
│ HTTP / WebSocket                     │
└───────────────┬──────────────────────┘
                │
                │ 网络
                ▼
          Lite3 / Robot
                │
                ▼
             Speaker
```

例如 HTTP 接口：

```http
POST /tts
Content-Type: application/json
```

请求：

```json
{
    "text": "你好，我是Lite3机器人。",
    "voice": "default"
}
```

服务端：

```text
JSON
 ↓
TTS Server
 ↓
VoxCPM2
 ↓
PCM / WAV
 ↓
返回客户端
```

如果要求低延迟，则进一步改为 WebSocket：

```text
Robot
  │
  │ WebSocket
  ▼
TTS Server
  │
  ▼
VoxCPM2
  │
  ├── PCM chunk 1
  ├── PCM chunk 2
  ├── PCM chunk 3
  └── ...
  │
  ▼
Robot Speaker
```

---

# 15. 推荐部署顺序

严格按照以下顺序操作：

```text
① 检查 Ubuntu 22.04
        ↓
② 检查 NVIDIA Driver
        ↓
③ nvidia-smi 正常
        ↓
④ 检查 Python
        ↓
⑤ 创建 venv
        ↓
⑥ 安装 PyTorch CUDA
        ↓
⑦ torch.cuda.is_available() == True
        ↓
⑧ 确认 GPU == RTX 4080
        ↓
⑨ pip install voxcpm
        ↓
⑩ 下载 VoxCPM2
        ↓
⑪ 最简单 TTS 测试
        ↓
⑫ 确认 GPU 推理
        ↓
⑬ 声音克隆测试
        ↓
⑭ Streaming 测试
        ↓
⑮ HTTP / WebSocket TTS Server
        ↓
⑯ 接入机器人
```

---

# 16. 第一阶段只需要执行这些命令

部署开始时，不要一次执行全部命令。

先执行：

```bash
cat /etc/os-release
```

```bash
uname -m
```

```bash
nvidia-smi
```

```bash
nvcc --version
```

```bash
python3 --version
```

然后建立环境：

```bash
mkdir -p ~/voxcpm
cd ~/voxcpm

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
```

接下来根据 `nvidia-smi` 和 Python 环境选择 PyTorch CUDA 版本。

---

# 17. PyTorch 安装后必须验证

执行：

```bash
python
```

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
```

预期：

```text
True
NVIDIA GeForce RTX 4080
(8, 9)
```

只有这里确认正常，才进入 VoxCPM 安装。

---

# 18. 常见问题

## 18.1 `nvidia-smi` 找不到 GPU

先解决 NVIDIA Driver。

不要继续安装 VoxCPM。

---

## 18.2 PyTorch 显示 CUDA False

例如：

```text
torch.cuda.is_available()
False
```

重点检查：

```bash
nvidia-smi
```

以及：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

通常是 PyTorch CUDA wheel 与当前 Driver / 环境不匹配。

---

## 18.3 GPU 显示 RTX 4080，但是 VoxCPM 使用 CPU

确认加载模型时使用：

```python
device="cuda"
```

并检查：

```python
torch.cuda.is_available()
```

---

## 18.4 Hugging Face 下载失败

可以：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

再重新运行。

---

## 18.5 显存不足

先确认：

```bash
nvidia-smi
```

查看其他进程是否占用显存。

RTX 4080 通常有足够的显存用于 VoxCPM 推理，但具体占用取决于模型版本、推理参数和同时运行的其他程序。

---

# 19. 后续建议

完成基础部署以后，再逐步增加：

```text
阶段 1
VoxCPM2 单机 TTS
        ↓
阶段 2
声音克隆
        ↓
阶段 3
Streaming
        ↓
阶段 4
HTTP TTS Server
        ↓
阶段 5
WebSocket Streaming Server
        ↓
阶段 6
Lite3 / Robot 接入
        ↓
阶段 7
开机自启动 systemd
        ↓
阶段 8
日志、监控、异常恢复
```

建议不要在第一阶段就加入 HTTP Server、WebSocket、systemd 等功能。先确保：

```text
RTX4080
  +
PyTorch CUDA
  +
VoxCPM2
  +
中文 TTS
```

完整运行正常，再逐步增加服务化功能。

---

# 20. 当前部署检查清单

## 系统

- [ ] Ubuntu 22.04
- [ ] x86_64
- [ ] RTX 4080
- [ ] NVIDIA Driver 正常
- [ ] `nvidia-smi` 正常
- [ ] CUDA 环境正常
- [ ] Python 3.10 / 3.11

## Python

- [ ] 创建 `~/voxcpm/.venv`
- [ ] 激活 `.venv`
- [ ] pip 更新
- [ ] PyTorch 安装
- [ ] `torch.cuda.is_available() == True`
- [ ] GPU 显示 RTX 4080
- [ ] Compute Capability `(8, 9)`

## VoxCPM

- [ ] `pip install voxcpm`
- [ ] `from voxcpm import VoxCPM`
- [ ] VoxCPM2 模型下载成功
- [ ] 中文 TTS 成功
- [ ] `test.wav` 生成成功
- [ ] GPU 显存正常增加
- [ ] 声音克隆测试
- [ ] Streaming 测试

## 服务化

- [ ] HTTP TTS Server
- [ ] WebSocket TTS Server
- [ ] Streaming PCM
- [ ] Robot / Lite3 接入
- [ ] systemd 自启动
- [ ] 日志与异常恢复

---

# 21. 本次部署的核心原则

**先验证 GPU，再安装模型；先跑通单机 TTS，再做服务化。**

最关键的验证点只有：

```bash
nvidia-smi
```

以及：

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

必须确认：

```text
True
NVIDIA GeForce RTX 4080
```

然后再继续 VoxCPM2。

---

## 22. 下一步

首次部署时，将以下命令的完整输出保存下来：

```bash
cat /etc/os-release
uname -m
nvidia-smi
nvcc --version
python3 --version
```

根据这些信息确定 PyTorch CUDA 版本后，再执行 VoxCPM2 安装。

后续建议最终整理成以下部署文件：

```text
voxcpm/
├── .venv/
├── models/
├── test_tts.py
├── test_voice_clone.py
├── tts_server.py
├── websocket_server.py
├── install_voxcpm.sh
├── start_voxcpm_server.sh
└── README.md
```

其中：

- `install_voxcpm.sh`：安装 Python 环境、PyTorch、VoxCPM
- `test_tts.py`：基础 TTS 测试
- `test_voice_clone.py`：声音克隆测试
- `tts_server.py`：HTTP TTS 服务
- `websocket_server.py`：Streaming TTS 服务
- `start_voxcpm_server.sh`：启动服务
- `README.md`：完整部署和使用说明

整体部署结构
┌──────────────────────────────────────────┐
│ Ubuntu 22.04                            │
│                                          │
│  NVIDIA Driver                           │
│       │                                  │
│       ▼                                  │
│  CUDA Runtime / PyTorch CUDA             │
│       │                                  │
│       ▼                                  │
│  Python 3.10/3.11                       │
│       │                                  │
│       ▼                                  │
│  VoxCPM                                  │
│       │                                  │
│       ├── VoxCPM2 模型                   │
│       │                                  │
│       ├── 文本 → 语音                    │
│       │                                  │
│       ├── 声音克隆                       │
│       │                                  │
│       └── Streaming                      │
│              │                           │
│              ▼                           │
│          WAV / PCM                       │
└──────────────────────────────────────────┘

接你的机器人控制系统，可以进一步变成：
机器人 / 上位机
      │
      │ HTTP / WebSocket / TCP
      ▼
VoxCPM TTS Server
      │
      ▼
VoxCPM2
      │
      ▼
PCM / WAV
      │
      ▼
音频播放 / 机器人扬声器



////////////////////Web Demo////////////////////////////////////////////////////////////////////////////////
# 1. Clone the repo (required even if you pip-installed voxcpm)
git clone https://github.com/OpenBMB/VoxCPM.git
cd VoxCPM

# 2. Install in editable mode from the source checkout
pip install -e .

# 3. Launch the web app
python app.py




