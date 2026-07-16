#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================"
echo "  quant-it — dependency installer"
echo "========================================"

# HF cache: Kaggle → /kaggle/working, local → ~/.cache
if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]; then
    export HF_HOME="/kaggle/working/.cache/huggingface"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE="$HF_HOME"

PYTHON="$(command -v python3)"

# --- check existing ---
HAS_TORCH=false
HAS_UNSLOTH=false
if $PYTHON -c "import torch; import transformers" 2>/dev/null; then
    HAS_TORCH=true
    echo "  torch + transformers already installed"
fi
if $PYTHON -c "import unsloth" 2>/dev/null; then
    HAS_UNSLOTH=true
    echo "  unsloth already installed"
fi

# --- decide: venv or system ---
USE_VENV=false
if [ "$HAS_TORCH" = false ]; then
    # try to install system-wide first; if that fails, fall back to venv
    if ! $PYTHON -m pip install --quiet torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
        USE_VENV=true
    fi
fi

if [ "$USE_VENV" = true ]; then
    echo "  system install failed, using virtual environment..."

    if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
        rm -rf .venv
    fi

    if [ ! -d ".venv" ]; then
        $PYTHON -m venv .venv
    fi

    source .venv/bin/activate
    PYTHON=".venv/bin/python3"

    $PYTHON -m pip install --upgrade pip -q
    $PYTHON -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
fi

# --- core deps ---
echo ""
echo "installing core dependencies..."
$PYTHON -m pip install transformers accelerate safetensors sentencepiece protobuf psutil -q

# --- unsloth (optional) ---
if [ "$HAS_UNSLOTH" = false ]; then
    echo "installing unsloth..."
    $PYTHON -m pip install unsloth -q 2>/dev/null \
        || $PYTHON -m pip install "unsloth[cu121]" -q 2>/dev/null \
        || {
            echo "  pip unsloth failed, trying from git..."
            $PYTHON -m pip install "git+https://github.com/unslothai/unsloth.git" -q 2>/dev/null \
                || echo "  unsloth install skipped (optional)"
        }
fi

# --- verify ---
echo ""
echo "verifying installation..."
$PYTHON -c "
import torch
print(f'  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
import transformers
print(f'  transformers {transformers.__version__}')
try:
    import unsloth; print(f'  unsloth {unsloth.__version__}')
except: print('  unsloth: not installed (optional)')
"

echo ""
echo "done — all dependencies installed"
echo ""
echo "usage:"
echo "  $PYTHON quantize.py --preset qwen35b"
echo "  $PYTHON quantize.py --download qwen35b    # pre-download model to cache"
echo "  $PYTHON quantize.py --download qwen05b"
echo "  $PYTHON quantize.py --preset qwen05b"
