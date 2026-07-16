#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=============================================="
echo "  Qwen3.6-35B-A3B (MoE) — quantize + finetune"
echo "=============================================="

# --- persistent HF cache for Kaggle ---
export HF_HOME="/kaggle/working/.cache/huggingface"
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
    transformers accelerate safetensors sentencepiece protobuf psutil
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
print(f'  Disk: {shutil.disk_usage(\".\").free/1024**3:.1f} GB free')
import shutil
" 2>/dev/null || true

# --- download model if not cached ---
echo ""
echo "[3/5] checking model cache..."
MODEL_CACHE="$HF_HOME/hub"
if [ -d "$MODEL_CACHE" ] && ls "$MODEL_CACHE"/models--Qwen--Qwen3.6-35B-A3B 2>/dev/null | head -1 > /dev/null; then
    echo "  Qwen3.6-35B-A3B already cached"
else
    echo "  downloading Qwen3.6-35B-A3B (~20 GB)..."
    $PYTHON quantize.py --download qwen35b
fi

# --- quantize (PTQ only — teacher won't fit alongside student on 2x T4) ---
echo ""
echo "[4/5] quantizing Qwen3.6-35B-A3B (ternary, PTQ mode)..."
echo "  note: teacher distillation skipped automatically"
echo "  (35B MoE + teacher don't fit in 2x T4 + 32GB RAM)"
echo ""
$PYTHON quantize.py --preset qwen35b \
    --device cuda:0 \
    --expert-batch 4 \
    --output Qwen3.6-35B-A3B-ternary

echo ""
echo "=============================================="
echo "  done!"
echo "=============================================="
echo ""
echo "outputs:"
echo "  packed:  Qwen3.6-35B-A3B-ternary/"
echo "  files:"
echo "    quantized_weights.pt   — bit-packed scales + codes"
echo "    quant_config.pt        — packing metadata"
echo "    config.json            — model config"
echo "    tokenizer*             — tokenizer files"
echo ""
echo "  this is the compressed model (~8-16x smaller than FP16)"
echo ""
echo "  to use for inference, load with:"
echo "    from quantize import unpack_q1_0, unpack_q2_0"
echo "    state = torch.load('quantized_weights.pt')"
echo "    config = torch.load('quant_config.pt')"
