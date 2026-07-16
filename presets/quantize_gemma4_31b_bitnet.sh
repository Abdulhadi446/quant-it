#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=============================================="
echo "  Gemma 4 31B (dense) — BitNet 1-bit"
echo "=============================================="
echo "  30.7B params, 60 layers, 256K context"
echo "  mode: binary {-1, +1}, 1.125 bits/weight"
echo "  expected size: ~4 GB GGUF"
echo "=============================================="

# --- persistent HF cache ---
if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]; then
    export HF_HOME="/kaggle/working/.cache/huggingface"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR"

PYTHON="$(command -v python3)"
echo "HF cache: $HF_HOME"

# --- install deps ---
echo ""
echo "[1/5] installing dependencies..."
$PYTHON -m pip install -q \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
$PYTHON -m pip install -q \
    transformers accelerate safetensors sentencepiece protobuf psutil gguf
$PYTHON -m pip install -q unsloth 2>/dev/null || \
    $PYTHON -m pip install -q "unsloth[cu121]" 2>/dev/null || \
    echo "  unsloth install skipped"

# --- check GPU ---
echo ""
echo "[2/5] hardware check..."
$PYTHON -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs: {n}')
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}  {free/1024**3:.1f}/{total/1024**3:.1f} GB')
try:
    import psutil
    m = psutil.virtual_memory()
    print(f'  RAM: {m.available/1024**3:.1f}/{m.total/1024**3:.1f} GB free')
except: pass
import shutil
print(f'  Disk: {shutil.disk_usage(\".\").free/1024**3:.1f} GB free')
" 2>/dev/null || true

# --- download model if not cached ---
echo ""
echo "[3/5] checking model cache..."
MODEL_CACHE="$HF_HOME/hub"
if [ -d "$MODEL_CACHE" ] && ls "$MODEL_CACHE"/models--google--gemma-4-31B 2>/dev/null | head -1 > /dev/null; then
    echo "  gemma-4-31B already cached"
else
    echo "  downloading google/gemma-4-31B (~60 GB)..."
    $PYTHON quantize.py --download gemma4_31b
fi

# --- quantize (BitNet 1-bit) ---
echo ""
echo "[4/5] quantizing Gemma 4 31B (BitNet 1-bit, PTQ mode)..."
echo "  weights: {-1, +1} per block, scale = max(|w|)"
echo "  dense model — no MoE expert batching needed"
echo ""
$PYTHON quantize.py --preset gemma4_31b \
    --mode 1bit \
    --device cuda:0 \
    --output gemma-4-31b-bitnet

# --- convert to GGUF ---
echo ""
echo "[5/5] converting to GGUF..."
$PYTHON quantize.py --gguf gemma-4-31b-bitnet \
    --output gemma-4-31b-bitnet.gguf

echo ""
echo "=============================================="
echo "  done!"
echo "=============================================="
echo ""
echo "output:"
echo "  GGUF: gemma-4-31b-bitnet.gguf"
echo ""
echo "use with llama.cpp:"
echo "  ./llama-server -m gemma-4-31b-bitnet.gguf -c 4096"
