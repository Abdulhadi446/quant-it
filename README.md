# quant-it

**1-bit and ternary quantization for any HuggingFace model. Auto-detects Mixture-of-Experts, streams expert batches across dual GPUs, and supports knowledge distillation. Built for consumer hardware.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-optimized-green.svg)](https://github.com/unslothai/unsloth)

---

## What This Does

Quantizes **any** HuggingFace transformer model to extreme low-bit formats:

| Mode | Weights | Bits per weight | Compression vs FP16 |
|------|---------|----------------|---------------------|
| **1-bit (Q1_0_g128)** | `{-1, +1}` binary | 1.125 | ~14× |
| **Ternary (Q2_0_g128)** | `{-1, 0, +1}` | 2.125 (1.58-bit info) | ~7× |

Both formats use **group-128 block quantization** with per-group FP16 scales — matching the Q1_0_g128 / Q2_0 formats used by the [Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) family and llama.cpp's GGUF.

Inspired by the [Bonsai](https://github.com/deepgrove-ai/Bonsai) (deepgrove) and [PrismML](https://github.com/PrismML-Eng/Bonsai-demo) research on ultra-efficient language models.

See [paper.md](paper.md) for the full technical description of the MoE quantization approach.

---

## Key Features

### MoE-Aware Quantization

Automatic detection of Mixture-of-Experts architectures (Qwen-MoE, Mixtral, DeepSeek, DBRX, Arctic, etc.).

**Router/gate layers are never quantized** — only expert feed-forward weights are binarized or ternarized. This is critical: quantizing gate layers causes routing collapse and immediate quality degradation. The gate layers (`gate_proj`, `router`, `mlp.gate`, `block_sparse_moe.gate`) account for < 0.1% of total parameters and are preserved in FP16.

Supported MoE router patterns:

| Architecture | Router Detected | Expert Pattern |
|---|---|---|
| Qwen3-MoE (30B-A3B, 35B-A3B) | `mlp.gate` | `mlp.experts.{i}` |
| Mixtral-8x7B / 8x22B | `block_sparse_moe.gate` | `block_sparse_moe.experts.{i}` |
| DeepSeek-V2 / V3 | `mlp.gate` | `mlp.experts.{i}` |
| DBRX | `ffn.router` | `ffn.experts.{i}` |

### Batched Expert Streaming

Fits massive MoE models on limited VRAM by loading experts in optimally-sized batches:

- **Auto mode**: measures free VRAM, calculates how many experts fit, processes them in batches
- **Manual mode**: you pick the batch size (e.g., 4, 8, 16 experts per iteration)
- **Pattern**: load batch → quantize in-place → offload to CPU → free cache → repeat

Example with Qwen3-30B-A3B (128 experts, 2×T4):
```
auto batch size: 56 experts/iter (~300 MB each, 0.6 × 28 GB / 300 MB)
layer 1/24: batch [0:56]    → quantize → offload
layer 1/24: batch [56:112]  → quantize → offload
layer 1/24: batch [112:128] → quantize → offload
...
```

### Knowledge Distillation

Use a full-precision teacher model to guide quantization with KL-divergence loss:

```bash
# Ternary quantize Llama-3-8B with 70B teacher
python quantize.py --model meta-llama/Llama-3-8B --teacher meta-llama/Llama-3-70B --mode ternary
```

Uses the **Straight-Through Estimator (STE)** for gradient flow through the quantization function. Router weights remain frozen throughout distillation.

### Pure PTQ

No teacher needed — post-training quantization with per-group symmetric scaling. Works well for ternary; 1-bit benefits from distillation.

---

## How It Works

### Quantization Functions

**1-bit (Q1_0_g128)**: Each group of 128 weights is replaced by its sign scaled by the group's mean absolute value:

```
scale = mean(|w|)  per 128-weight group
w_hat = scale * sign(w)    → {-scale, +scale}
```

**Ternary (Q2_0_g128)**: Each group of 128 weights is mapped to {−1, 0, +1} scaled by the group max:

```
scale = max(|w|)  per 128-weight group
q = clamp(round(w / scale), -1, 1)   → {-1, 0, +1}
w_hat = scale * q                     → {-scale, 0, +scale}
```

Ternary sets ~30% of weights to exactly zero — creating **double sparsity** on top of MoE's already-sparse activation pattern.

### MoE Router Detection

```python
SKIP_NAMES = {"gate_proj", "router", "router_proj", "mlp.gate", "block_sparse_moe.gate"}
```

Router weights matching these patterns are **never touched**. Only `nn.Linear` layers with "expert" in their module path (and not matching a router pattern) are quantized.

### Expert Batch Sizing

```
batch_size = floor( free_VRAM × 0.6 / bytes_per_expert )
```

The 0.6 safety factor accounts for CUDA allocator overhead and activation memory. With 2×T4 (~28 GB free) and 300 MB experts: `28 × 0.6 / 0.3 = 56 experts per batch`.

### Weight Packing

After quantization, weights are packed into compact int32 representations:

- **Q1_0_g128**: 32 sign bits per int32 word + 1 FP16 scale per 128 weights → ~1.125 bits/weight
- **Q2_0_g128**: 16 ternary codes (2 bits each) per int32 word + 1 FP16 scale per 128 weights → ~2.125 bits/weight

---

## Hardware Requirements

| Setup | Best For |
|-------|----------|
| **2×T4 (15.9 GB each) + 32 GB RAM** | MoE models up to 35B total params |
| **1×RTX 4090 (24 GB) + 64 GB RAM** | MoE up to 70B total, dense up to 30B |
| **1×T4/RTX 3090 (16–24 GB) + 32 GB RAM** | Dense models up to 8B; MoE up to 20B |
| **CPU only** | Small models (< 3B) |

Unsloth handles automatic GPU↔CPU offloading. Expert batching keeps peak VRAM independent of total model size.

---

## Quick Start

```bash
# clone the repo
git clone https://github.com/Abdulhadi446/quant-it.git
cd quant-it

# run the installer (creates venv, installs torch + unsloth + transformers)
./quant.sh
```

The interactive wizard will prompt you for:
1. **Preset** — choose from pre-configured models (Qwen3.6-35B-A3B, Llama-3-8B, etc.)
2. **Or custom** — manual configuration
3. Student model (any HF model name or local path)
4. Teacher model (optional — enables distillation)
5. Quantization mode (1-bit or ternary)
6. Expert batch size (auto or manual)
7. Output directory

### Preset Scripts (one-shot)

Ready-to-run scripts in `presets/` for popular models:

```bash
# Qwen3.6-35B-A3B — BitNet 1-bit (binary, ~4 GB GGUF)
bash presets/quantize_qwen35b_bitnet.sh

# Gemma 4 31B — dense, ternary (~7 GB GGUF)
bash presets/quantize_gemma4_31b.sh

# Gemma 4 31B — dense, 1-bit (~4 GB GGUF)
bash presets/quantize_gemma4_31b_bitnet.sh

# Gemma 4 26B-A4B — MoE, ternary (~7 GB GGUF)
bash presets/quantize_gemma4_26b.sh

# Gemma 4 26B-A4B — MoE, 1-bit (~4 GB GGUF)
bash presets/quantize_gemma4_26b_bitnet.sh

# Qwen3.5-0.8B — small dense, ternary (~200 MB GGUF)
bash presets/quantize_qwen35_08b.sh

# Qwen3.5-0.8B — small dense, 1-bit (~100 MB GGUF)
bash presets/quantize_qwen35_08b_bitnet.sh
```

Each script handles: install deps → download model → quantize → export GGUF.

---

## Usage

### Interactive Mode (recommended)

```bash
./quant.sh
```

### Step-by-Step Examples

#### 1. Ternary quantize a small dense model (fits on any GPU)

```bash
./quant.sh
# → student: Qwen/Qwen2.5-0.5B
# → teacher: (empty — PTQ)
# → mode: ternary
# → batch: auto
# → output: Qwen2.5-0.5B-ternary
```

#### 2. 1-bit quantize Llama with teacher distillation

```bash
./quant.sh
# → student: meta-llama/Llama-3-8B
# → teacher: meta-llama/Llama-3-70B
# → mode: 1bit
# → epochs: 3
# → device: cuda:0
# → calib file: (empty for dummy)
# → output: Llama-3-8B-1bit
```

#### 3. Ternary quantize a MoE model on 2×T4

```bash
./quant.sh
# → student: Qwen/Qwen3-30B-A3B
# → teacher: (empty — PTQ)
# → mode: ternary
# → batch: auto (fits ~56 experts per batch on 2×T4)
# → output: Qwen3-30B-A3B-ternary
```

#### 4. Manual batch size for MoE (4 experts at a time)

```bash
./quant.sh
# → student: Qwen/Qwen3-30B-A3B
# → teacher: (empty)
# → mode: ternary
# → batch: m → 4
# → output: Qwen3-30B-A3B-ternary
```

#### 5. 1-bit quantize Qwen3.6-35B-A3B (MoE)

```bash
python quantize.py --preset qwen35b
```

#### 6. Ternary quantize Mistral-7B

```bash
./quant.sh
# → student: mistralai/Mistral-7B-v0.1
# → teacher: (empty)
# → mode: ternary
# → output: Mistral-7B-v0.1-ternary
```

#### 7. 1-bit quantize Phi-3 with custom calibration data

```bash
# first create a calibration file
echo "The capital of France is Paris." > calib.txt
echo "Machine learning is a subset of AI." >> calib.txt
echo "Python is a popular programming language." >> calib.txt

python quantize.py --model microsoft/Phi-3-mini-4k-instruct --mode 1bit --calib-file calib.txt
```

#### 8. Quantize a local model

```bash
python quantize.py --model /path/to/my/local/model --mode ternary --output my-local-model-ternary
```

#### 9. Full distillation with custom hyperparameters

```bash
python quantize.py \
  --model Qwen/Qwen2.5-7B \
  --teacher Qwen/Qwen2.5-72B \
  --mode ternary \
  --epochs 5 \
  --lr 1e-4 \
  --batch-size 2 \
  --max-len 1024 \
  --calib-file calib.txt \
  --output Qwen2.5-7B-ternary-distilled
```

#### 10. Dequantize back to FP16 (for use with standard inference stacks)

```bash
python quantize.py --dequantize Qwen3-30B-A3B-ternary --output Qwen3-30B-A3B-fp16
```

### Programmatic Usage

```python
from quantize import ptq, ternary_weight, onebit_weight, quantize_experts_batched, is_moe
from transformers import AutoModelForCausalLM

# ternary PTQ — any model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = ptq(model, ternary_weight)
model.save_pretrained("qwen2.5-0.5b-ternary")

# 1-bit PTQ — dense model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = ptq(model, onebit_weight)
model.save_pretrained("llama3-8b-1bit")

# MoE-aware batched PTQ
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map="auto")
if is_moe(model):
    model = quantize_experts_batched(model, ternary_weight, dtype=torch.float16, device="cuda:0")
else:
    model = ptq(model, ternary_weight)
```

### Running from Jupyter / Kaggle

Shell commands need `!` prefix. Use `--model` or `--preset` flags (no interactive prompts):

```python
# quantize with preset (non-interactive)
!python quantize.py --preset qwen35b

# with teacher distillation
!python quantize.py \
  --model meta-llama/Llama-3-8B \
  --teacher meta-llama/Llama-3-70B \
  --mode ternary \
  --epochs 3 \
  --device cuda:0

# MoE model with custom expert batch size
!python quantize.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --mode ternary \
  --expert-batch 8 \
  --output qwen35b-ternary

# list available presets
!python quantize.py --preset list
```

### One-Liner Examples

```bash
# clone + install + run in one shot
git clone https://github.com/Abdulhadi446/quant-it.git && cd quant-it && ./quant.sh

# re-run after initial install (skips venv creation)
source .venv/bin/activate && python quantize.py

# quantize and push to HuggingFace
python quantize.py --model Qwen/Qwen2.5-0.5B --mode ternary --output qwen-ternary
huggingface-cli upload my-username/qwen-2.5-0.5b-ternary qwen-ternary
```

---

## Output

Quantized models are saved in a hybrid format alongside the standard HuggingFace config/tokenizer:

```
model_name-ternary/
├── config.json               # HuggingFace model config
├── tokenizer.json            # tokenizer
├── tokenizer_config.json
├── quantized_weights.pt      # packed {int32 bit-codes + FP16 scales}
└── quant_config.pt           # {pack_config, mode, group_size}
```

To use the output with standard HuggingFace inference, first dequantize to FP16 safetensors:

```bash
python quantize.py --dequantize model_name-ternary --output model_name-fp16
```

Then load as normal:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("llama3-8b-ternary-fp16")
tokenizer = AutoTokenizer.from_pretrained("llama3-8b-ternary-fp16")
```

---

## Benchmarks (Bonsai Reference)

The Bonsai family demonstrates what extreme quantization achieves on dense models. MoE models at comparable activated-parameter counts are expected to show similar quality retention.

| Model | Params | Format | Size | MMLU | Avg Benchmark |
|-------|--------|--------|------|------|--------------|
| Qwen3-8B FP16 | 8B | FP16 | 16.4 GB | 83.0 | 79.3 |
| Ternary Bonsai 8B | 8B | Q2_0_g128 | 2.16 GB | — | 75.5 (−4.6%) |
| 1-bit Bonsai 8B | 8B | Q1_0_g128 | 1.15 GB | 65.7 | 59.9 (−24%) |
| Qwen3.6-27B FP16 | 27B | FP16 | 54 GB | 93.4 | 85.1 |
| Ternary Bonsai 27B | 27B | Q2_0 | 5.9 GB | 88.1 | 80.5 (−5.4%) |
| 1-bit Bonsai 27B | 27B | Q1_0 | 3.9 GB | 82.8 | 76.1 (−10.6%) |
| Bonsai 0.5B (ternary) | 0.5B | Q2_0 | ~170 MB | 30.3 | — |

### Supported Presets

| Preset Script | Model | Type | Mode | Expected GGUF Size |
|--------------|-------|------|------|-------------------|
| `quantize_qwen35b_bitnet.sh` | Qwen3.6-35B-A3B | MoE | 1-bit | ~4 GB |
| `quantize_gemma4_31b.sh` | Gemma 4 31B | Dense | Ternary | ~7 GB |
| `quantize_gemma4_31b_bitnet.sh` | Gemma 4 31B | Dense | 1-bit | ~4 GB |
| `quantize_gemma4_26b.sh` | Gemma 4 26B-A4B | MoE | Ternary | ~7 GB |
| `quantize_gemma4_26b_bitnet.sh` | Gemma 4 26B-A4B | MoE | 1-bit | ~4 GB |
| `quantize_qwen35_08b.sh` | Qwen3.5-0.8B | Dense | Ternary | ~200 MB |
| `quantize_qwen35_08b_bitnet.sh` | Qwen3.5-0.8B | Dense | 1-bit | ~100 MB |

**Intelligence Density** = −log₂(Pₑ) / N_GB where Pₑ = 1 − score/100. Ternary models achieve 7–11× higher intelligence density than their FP16 counterparts.

---

## Technical Paper

See [paper.md](paper.md) for the full technical description of:
- Q1_0_g128 (binary) and Q2_0_g128 (true ternary) quantization mathematics
- Why router preservation is essential for MoE models
- Expert batch streaming algorithm and VRAM budget calculation
- Knowledge distillation with STE for MoE models
- Weight packing format specification
- Limitations and future work (QAT, native kernels, KV-cache quantization)

---

## Quantization Research References

This tool implements techniques from:

- **[Bonsai](https://github.com/deepgrove-ai/Bonsai)** — 500M ternary-weight LM trained from scratch (deepgrove-ai)
- **[PrismML Bonsai](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit and ternary models from 1.7B to 27B parameters
- **[1-bit Bonsai 8B](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)** — Q1_0_g128 binary quantization, 14× compression
- **[Ternary Bonsai 8B](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)** — True {−1, 0, +1} Q2_0_g128 at 8B/4B/1.7B scale
- **[Bonsai 27B](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)** — 27B reasoning in 3.9–5.9 GB; first 27B model on a phone
- **[Bonsai Image 4B](https://github.com/PrismML-Eng/Bonsai-Image-Demo/blob/main/bonsai-image-4b-whitepaper.pdf)** — Binary/ternary diffusion transformers

### Key Technical Insights Applied

| Concept | Source | Implementation |
|---------|--------|----------------|
| Group-128 symmetric scaling | Bonsai Q1_0_g128 / Q2_0 | `scale = mean/max(|w|)` per 128-weight group |
| True ternary {−1, 0, +1} | PrismML Ternary Bonsai | `clamp(round(w/scale), -1, 1)` |
| MoE router preservation | Bonsai 27B MoE analysis | skip `gate_proj`, `router` etc. layers |
| Expert streaming | Hardware constraints | batch-load, quantize, offload to CPU |
| STE for distillation | Standard QAT literature | straight-through estimator |
| Intelligence density metric | PrismML whitepapers | −log₂(Pₑ) / N_GB |

---

## Model Compatibility

Works with any HuggingFace `AutoModelForCausalLM`:

- **Dense**: Llama, Qwen, Mistral, Phi, Gemma, GPT-2, etc.
- **MoE**: Qwen-MoE, Mixtral, DeepSeek-V2/V3, DBRX, Arctic, etc.
- **Custom**: any model with `trust_remote_code=True`

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended; CPU-only works for small models)
- 16 GB+ system RAM (32 GB+ for large MoE models)

### Manual

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors sentencepiece protobuf psutil
pip install unsloth
```

### Automatic

```bash
./quant.sh  # creates venv, installs everything, launches interactive mode
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [deepgrove-ai](https://github.com/deepgrove-ai/Bonsai) for the original Bonsai ternary model
- [PrismML](https://github.com/PrismML-Eng/Bonsai-demo) for the Bonsai 1-bit, ternary, image, and 27B family
- [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient model loading and training
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for Q1_0_g128 and Q2_0 quantization format specification
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the model ecosystem
