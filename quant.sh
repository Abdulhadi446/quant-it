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

# --- check if packages are already installed system-wide ---
SKIP_VENV=false
if "$PYTHON" -c "import torch; import transformers" 2>/dev/null; then
    echo "  torch + transformers already installed, skipping venv"
    SKIP_VENV=true
fi

# --- venv ---
if [ "$SKIP_VENV" = false ]; then
    # remove broken venv if activate is missing
    if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
        echo "  removing broken .venv..."
        rm -rf .venv
    fi

    if [ ! -d ".venv" ]; then
        echo "[1/4] creating virtual environment..."
        "$PYTHON" -m venv .venv --without-pip 2>/dev/null || "$PYTHON" -m venv .venv
        # bootstrap pip if ensurepip was skipped
        if [ ! -f ".venv/bin/pip" ]; then
            curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3
        fi
    else
        echo "[1/4] venv already exists"
    fi

    source .venv/bin/activate

    # --- pip upgrade ---
    echo "[2/4] upgrading pip..."
    pip install --upgrade pip -q 2>/dev/null || true

    # --- torch (CUDA 12.1 for T4) ---
    echo "[3/4] installing torch + unsloth..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null \
        || pip install torch torchvision torchaudio -q 2>/dev/null \
        || echo "  torch install skipped (already present or no CUDA)"

    # Unsloth — install from pip (prebuilt wheels)
    pip install unsloth -q 2>/dev/null \
        || pip install "unsloth[cu121]" -q 2>/dev/null \
        || {
            echo "  pip install failed, installing from git..."
            pip install "git+https://github.com/unslothai/unsloth.git" -q 2>/dev/null \
                || echo "  unsloth install skipped"
        }

    # transformers ecosystem
    pip install transformers accelerate safetensors sentencepiece protobuf psutil -q 2>/dev/null || true
else
    echo "[1/4] venv skipped (packages found)"
    echo "[2/4] pip skipped"
    echo "[3/4] torch + unsloth skipped"
fi

# --- hardware check ---
echo "[4/4] hardware check..."
"$PYTHON" -c "
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
" 2>/dev/null || echo "  (hardware check skipped)"

echo ""
echo "launching quantizer..."
echo ""
"$PYTHON" quantize.py
