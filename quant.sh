#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================"
echo "  quant-it installer + launcher"
echo "  (Unsloth + dual T4 optimized)"
echo "========================================"

# --- python ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

PYTHON="$(command -v python3)"

# --- venv ---
if [ ! -d ".venv" ]; then
    echo "[1/4] creating virtual environment..."
    "$PYTHON" -m venv .venv
else
    echo "[1/4] venv already exists"
fi

source .venv/bin/activate

# --- pip upgrade ---
echo "[2/4] upgrading pip..."
pip install --upgrade pip -q

# --- torch (CUDA 12.1 for T4) ---
echo "[3/4] installing torch + unsloth..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null \
    || pip install torch torchvision torchaudio -q

# Unsloth — install from pip (prebuilt wheels)
pip install unsloth -q 2>/dev/null \
    || pip install "unsloth[cu121]" -q 2>/dev/null \
    || {
        echo "  pip install failed, installing from git..."
        pip install "git+https://github.com/unslothai/unsloth.git" -q
    }

# transformers ecosystem
pip install transformers accelerate safetensors sentencepiece protobuf psutil -q

# --- hardware check ---
echo "[4/4] hardware check..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs found: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'    [{i}] {name}  {free/1024**3:.1f}/{total/1024**3:.1f} GB')
try:
    import psutil
    m = psutil.virtual_memory()
    print(f'  CPU RAM: {m.available/1024**3:.1f}/{m.total/1024**3:.1f} GB free')
except:
    pass
"

echo ""
echo "launching quantizer..."
echo ""
python3 quantize.py
