#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=============================================="
echo "  Qwen3.6-35B-A3B (MoE) — BitNet 1-bit"
echo "=============================================="
echo "  mode: binary {-1, +1}, 1.125 bits/weight"
echo "  expected size: ~4 GB GGUF"
echo "=============================================="

# --- HF cache on /dev/shm (RAM-backed, no disk usage) ---
if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]; then
    export HF_HOME="/dev/shm/hf_cache"
    WORK_DIR="/tmp/quant_work"
else
    export HF_HOME="$HOME/.cache/huggingface"
    WORK_DIR="."
fi
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$WORK_DIR"

PYTHON="$(command -v python3)"
echo "HF cache: $HF_HOME"
echo "Work dir: $WORK_DIR"

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
print(f'  Disk: {shutil.disk_usage(\"/tmp\").free/1024**3:.1f} GB free (/tmp)')
" 2>/dev/null || true

# --- download model if not cached ---
echo ""
echo "[3/5] checking model cache..."
MODEL_CACHE="$HF_HOME/hub"
if [ -d "$MODEL_CACHE" ] && ls "$MODEL_CACHE"/models--Qwen--Qwen3.6-35B-A3B 2>/dev/null | head -1 > /dev/null; then
    echo "  Qwen3.6-35B-A3B already cached"
else
    echo "  downloading Qwen3.6-35B-A3B (~35 GB) to $HF_HOME..."
    $PYTHON quantize.py --download qwen35b
fi

# --- quantize (BitNet 1-bit) ---
echo ""
echo "[4/5] quantizing (BitNet 1-bit, PTQ mode)..."
echo "  weights: {-1, +1} per block, scale = max(|w|)"
echo "  note: teacher distillation skipped — 35B MoE + teacher won't fit"
echo ""
$PYTHON quantize.py --preset qwen35b \
    --mode 1bit \
    --device cuda:0 \
    --expert-batch 4 \
    --output "$WORK_DIR/Qwen3.6-35B-A3B-bitnet"

# --- convert to GGUF ---
echo ""
echo "[5/5] converting to GGUF..."
$PYTHON quantize.py --gguf "$WORK_DIR/Qwen3.6-35B-A3B-bitnet" \
    --output "$WORK_DIR/Qwen3.6-35B-A3B-bitnet.gguf"

echo ""
echo "=============================================="
echo "  done!"
echo "=============================================="
echo ""
echo "output:"
echo "  GGUF: $WORK_DIR/Qwen3.6-35B-A3B-bitnet.gguf"
echo ""
echo "use with llama.cpp:"
echo "  ./llama-server -m $WORK_DIR/Qwen3.6-35B-A3B-bitnet.gguf -c 4096"
