#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=============================================="
echo "  Gemma 4 26B-A4B (MoE) — quantize + GGUF"
echo "=============================================="
echo "  25.2B total, 3.8B active, 128 experts"
echo "  mode: ternary {-1, 0, +1}, 1.71 bits/weight"
echo "  expected size: ~7 GB GGUF"
echo "=============================================="

# --- HF cache on /tmp (more disk space on Kaggle) ---
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
if [ -d "$MODEL_CACHE" ] && ls "$MODEL_CACHE"/models--google--gemma-4-26B-A4B 2>/dev/null | head -1 > /dev/null; then
    echo "  gemma-4-26B-A4B already cached"
else
    echo "  downloading google/gemma-4-26B-A4B (~50 GB) to $HF_HOME..."
    $PYTHON quantize.py --download gemma4_26b
fi

# --- quantize (ternary) ---
echo ""
echo "[4/5] quantizing Gemma 4 26B-A4B (ternary, PTQ mode)..."
echo "  weights: {-1, 0, +1} per block, scale = max(|w|)"
echo "  MoE model — expert/router layers kept in fp16"
echo ""
$PYTHON quantize.py --preset gemma4_26b \
    --mode ternary \
    --device cuda:0 \
    --expert-batch 8 \
    --output "$WORK_DIR/gemma-4-26b-A4B-ternary"

# --- convert to GGUF ---
echo ""
echo "[5/5] converting to GGUF..."
$PYTHON quantize.py --gguf "$WORK_DIR/gemma-4-26b-A4B-ternary" \
    --output "$WORK_DIR/gemma-4-26b-A4B-ternary.gguf"

echo ""
echo "=============================================="
echo "  done!"
echo "=============================================="
echo ""
echo "output:"
echo "  GGUF: $WORK_DIR/gemma-4-26b-A4B-ternary.gguf"
echo ""
echo "use with llama.cpp:"
echo "  ./llama-server -m $WORK_DIR/gemma-4-26b-A4B-ternary.gguf -c 4096"
