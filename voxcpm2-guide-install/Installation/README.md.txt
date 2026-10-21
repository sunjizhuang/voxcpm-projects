虑到你已经确定保留 Python 3.14.6 + 当前 .
venv，我建议把部署脚本设计成“不重建环境、自动检查、分阶段验证”的方式。


下面这套文件放在：

~/voxcpm/
├── .venv/
├── install_voxcpm.sh
├── test_voxcpm.py
└── start_voxcpm_server.sh

其中：

install_voxcpm.sh：检查环境 + 安装 PyTorch + VoxCPM
test_voxcpm.py：检查 GPU + 加载 VoxCPM2 + 生成 WAV
start_voxcpm_server.sh：启动一个简单 HTTP TTS 服务
1. install_voxcpm.sh
#!/usr/bin/env bash

set -e

VOXCPM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VOXCPM_DIR}/.venv"

echo "========================================"
echo " VoxCPM Installation"
echo " Ubuntu 22.04 + RTX 4080"
echo "========================================"

echo
echo "[1/7] Check Python environment"

if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: .venv not found:"
    echo "  ${VENV_DIR}"
    exit 1
fi

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found:"
    echo "  ${PYTHON}"
    exit 1
fi

echo "Python:"
"${PYTHON}" --version

echo
echo "Python executable:"
echo "${PYTHON}"

echo
echo "[2/7] Check NVIDIA GPU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader

echo
echo "[3/7] Upgrade pip"

"${PYTHON}" -m pip install -U pip setuptools wheel

echo
echo "[4/7] Install PyTorch CUDA"

"${PIP}" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

echo
echo "[5/7] Verify PyTorch CUDA"

"${PYTHON}" - <<'PY'
import torch
import sys

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print()
    print("ERROR: PyTorch cannot access CUDA.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

if torch.cuda.get_device_capability(0) != (8, 9):
    print("WARNING: GPU capability is not (8, 9).")

PY

echo
echo "[6/7] Install VoxCPM"

"${PIP}" install voxcpm

echo
echo "[7/7] Verify VoxCPM"

"${PYTHON}" - <<'PY'
from voxcpm import VoxCPM

print("VoxCPM Python package: OK")
print("VoxCPM class:", VoxCPM)
PY

echo
echo "========================================"
echo " Installation completed"
echo "========================================"

echo
echo "Next:"
echo "  ${VOXCPM_DIR}/.venv/bin/python ${VOXCPM_DIR}/test_voxcpm.py"
echo
2. test_voxcpm.py

这个脚本负责真正验证：

Python
 ↓
PyTorch
 ↓
CUDA
 ↓
RTX 4080
 ↓
VoxCPM2
 ↓
TTS
 ↓
test.wav
#!/usr/bin/env python3

import os
import sys
import time

import torch
import soundfile as sf

from voxcpm import VoxCPM


MODEL_ID = "openbmb/VoxCPM2"
OUTPUT_FILE = "test.wav"

TEXT = "你好，这是一段运行在 RTX 4080 上的 VoxCPM2 中文语音测试。"


def print_system_info():
    print("=" * 60)
    print("VoxCPM2 Test")
    print("=" * 60)

    print("Python:")
    print(sys.version)

    print()
    print("PyTorch:")
    print(torch.__version__)

    print()
    print("PyTorch CUDA:")
    print(torch.version.cuda)

    print()
    print("CUDA available:")
    print(torch.cuda.is_available())

    if not torch.cuda.is_available():
        print()
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    print()
    print("GPU:")
    print(torch.cuda.get_device_name(0))

    print()
    print("GPU capability:")
    print(torch.cuda.get_device_capability(0))

    print()
    print("GPU memory:")
    props = torch.cuda.get_device_properties(0)
    print(f"{props.total_memory / 1024**3:.2f} GB")


def main():
    print_system_info()

    print()
    print("=" * 60)
    print("Loading VoxCPM2")
    print("=" * 60)

    start = time.time()

    model = VoxCPM.from_pretrained(
        MODEL_ID,
        device="cuda",
        load_denoiser=False,
    )

    print(f"Model loaded in {time.time() - start:.2f} seconds")

    print()
    print("=" * 60)
    print("Generate speech")
    print("=" * 60)

    print("Text:")
    print(TEXT)

    start = time.time()

    wav = model.generate(
        text=TEXT,
        cfg_value=2.0,
        inference_timesteps=10,
    )

    elapsed = time.time() - start

    sf.write(
        OUTPUT_FILE,
        wav,
        model.tts_model.sample_rate,
    )

    print()
    print(f"Generation time: {elapsed:.2f} seconds")
    print(f"Sample rate: {model.tts_model.sample_rate}")
    print(f"Output: {os.path.abspath(OUTPUT_FILE)}")

    print()
    print("=" * 60)
    print("VoxCPM2 TTS TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
3. start_voxcpm_server.sh

这里先做一个最小可用 HTTP 服务，不要一开始把 WebSocket、Streaming、systemd 全部塞进去。

#!/usr/bin/env bash

set -e

VOXCPM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${VOXCPM_DIR}/.venv/bin/python"

SERVER="${VOXCPM_DIR}/tts_server.py"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python environment not found:"
    echo "${PYTHON}"
    exit 1
fi

if [ ! -f "${SERVER}" ]; then
    echo "ERROR: tts_server.py not found:"
    echo "${SERVER}"
    exit 1
fi

echo "========================================"
echo " Starting VoxCPM2 TTS Server"
echo "========================================"

echo "Python:"
"${PYTHON}" --version

echo
echo "Server:"
echo "${SERVER}"

exec "${PYTHON}" "${SERVER}"

不过这个启动脚本需要配套一个 tts_server.py。

4. tts_server.py

建议也一起放到 ~/voxcpm/：

#!/usr/bin/env python3

import io
import time

import soundfile as sf
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from voxcpm import VoxCPM


MODEL_ID = "openbmb/VoxCPM2"

app = FastAPI(
    title="VoxCPM2 TTS Server",
    version="1.0.0",
)


print("=" * 60)
print("Loading VoxCPM2")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

model = VoxCPM.from_pretrained(
    MODEL_ID,
    device="cuda",
    load_denoiser=False,
)

print("VoxCPM2 loaded")


class TTSRequest(BaseModel):
    text: str
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
    }


@app.post("/tts")
def tts(request: TTSRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="text is empty",
        )

    start = time.time()

    wav = model.generate(
        text=request.text,
        cfg_value=request.cfg_value,
        inference_timesteps=request.inference_timesteps,
    )

    elapsed = time.time() - start

    buffer = io.BytesIO()

    sf.write(
        buffer,
        wav,
        model.tts_model.sample_rate,
        format="WAV",
    )

    audio = buffer.getvalue()

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": f"{elapsed:.3f}",
            "X-Sample-Rate": str(model.tts_model.sample_rate),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
5. 安装 Server 依赖

在 .venv 中执行：

cd ~/voxcpm
source .venv/bin/activate

安装：

python -m pip install fastapi uvicorn soundfile
6. 给脚本执行权限
cd ~/voxcpm

chmod +x install_voxcpm.sh
chmod +x test_voxcpm.py
chmod +x start_voxcpm_server.sh

目录最终：

~/voxcpm/
├── .venv/
├── install_voxcpm.sh
├── test_voxcpm.py
├── tts_server.py
└── start_voxcpm_server.sh
7. 正式开始安装

你现在已经有：

Python 3.14.6
.venv
RTX 4080
Driver 580.178.04
CUDA Driver 13.0

因此：

cd ~/voxcpm
./install_voxcpm.sh

脚本会依次：

检查 .venv
    ↓
检查 RTX 4080
    ↓
升级 pip
    ↓
安装 PyTorch CUDA
    ↓
验证 CUDA
    ↓
安装 VoxCPM
    ↓
验证 VoxCPM

如果 PyTorch 在 Python 3.14 下没有对应 wheel，脚本会在 PyTorch 这一步停止。

这时候不要改 Python，我们再针对 Python 3.14 处理。

8. 安装完成后测试

执行：

cd ~/voxcpm
./test_voxcpm.py

第一次执行会下载：

openbmb/VoxCPM2

然后生成：

~/voxcpm/test.wav

可以检查：

ls -lh test.wav

播放：

aplay test.wav

同时建议另开终端：

watch -n 0.5 nvidia-smi

观察 RTX 4080 显存和 GPU 利用率。

9. 启动 HTTP TTS Server

安装完依赖以后：

cd ~/voxcpm
./start_voxcpm_server.sh

正常：

============================================================
Loading VoxCPM2
============================================================
GPU: NVIDIA GeForce RTX 4080
CUDA: 12.6
VoxCPM2 loaded

服务：

0.0.0.0:8000
10. 测试服务

另开终端：

curl http://127.0.0.1:8000/health

应该返回类似：

{
    "status": "ok",
    "model": "openbmb/VoxCPM2",
    "gpu": "NVIDIA GeForce RTX 4080",
    "cuda": "12.6"
}

然后测试 TTS：

curl -X POST \
    http://127.0.0.1:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"你好，我是运行在RTX 4080上的VoxCPM机器人语音系统。"}' \
    --output output.wav

检查：

ls -lh output.wav

播放：

aplay output.wav
11. 最终建议的部署路径

你的实际部署建议按照：

~/voxcpm
│
├── .venv
│
├── install_voxcpm.sh
│       │
│       ├── PyTorch
│       └── VoxCPM2
│
├── test_voxcpm.py
│       │
│       └── GPU + VoxCPM2 + TTS
│
├── tts_server.py
│       │
│       └── HTTP TTS
│
└── start_voxcpm_server.sh
        │
        └── 启动 TTS Server

部署顺序：

./install_voxcpm.sh
        ↓
./test_voxcpm.py
        ↓
确认 test.wav
        ↓
./start_voxcpm_server.sh
        ↓
curl /health
        ↓
curl /tts
        ↓
机器人接入
