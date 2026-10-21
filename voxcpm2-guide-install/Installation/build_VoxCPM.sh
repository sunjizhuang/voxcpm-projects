cd ~/voxcpm && cat > install_voxcpm.sh <<'EOF'
#!/usr/bin/env bash
set -e

VOXCPM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VOXCPM_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

echo "========================================"
echo " VoxCPM Installation"
echo " Ubuntu 22.04 + RTX 4080"
echo "========================================"

echo
echo "[1/7] Check Python environment"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found:"
    echo "  ${PYTHON}"
    exit 1
fi

"${PYTHON}" --version
echo "Python: ${PYTHON}"

echo
echo "[2/7] Check NVIDIA GPU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

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
    print("ERROR: PyTorch cannot access CUDA.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

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
echo "Run:"
echo "  ./test_voxcpm.py"
EOF

cat > test_voxcpm.py <<'EOF'
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
EOF

cat > tts_server.py <<'EOF'
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
EOF

cat > start_voxcpm_server.sh <<'EOF'
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
EOF

chmod +x install_voxcpm.sh test_voxcpm.py start_voxcpm_server.sh

echo
echo "========================================"
echo " VoxCPM scripts created"
echo "========================================"
ls -lh install_voxcpm.sh test_voxcpm.py tts_server.py start_voxcpm_server.sh
