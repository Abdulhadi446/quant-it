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
| **1-bit** | `{-1, +1}` binary | 1.0 | 16x |
| **Ternary** | `{-1, 0, +1}` | ~1.5-1.7 | ~10x |

Inspired by the [Bonsai](https://github.com/deepgrove-ai/Bonsai) (deepgrove) and [PrismML](https://github.com/PrismML-Eng/Bonsai-demo) research on ultra-efficient language models.

---

## Key Features

### MoE-Aware Quantization

Automatic detection of Mixture-of-Experts architectures (Qwen-MoE, Mixtral, DeepSeek, etc.). Router/gate layers are **never quantized** — only expert feed-forward weights are binarized or ternarized.

### Batched Expert Streaming

Fits massive MoE models on limited VRAM by loading experts in optimally-sized batches:

- **Auto mode**: measures free VRAM, calculates how many experts fit, processes them in parallel batches
- **Manual mode**: you pick the batch size (e.g., 4, 8, 16 experts per iteration)
- **Expert-by-expert**: load → quantize → offload to CPU → repeat

Example with Qwen3-30B-A3B (128 experts, 2x T4):
```
auto batch size: 50 experts/iter (~300 MB each)
layer 1/24: batch [0:50] → quantize → offload
layer 1/24: batch [50:100] → quantize → offload
layer 1/24: batch [100:128] → quantize → offload
```

### Knowledge Distillation

Use a full-precision teacher model to guide quantization with KL-divergence loss:

```bash
# Ternary quantize Llama-3-8B with 70B teacher
./quant.sh
# → student: meta-llama/Llama-3-8B
# → teacher: meta-llama/Llama-3-70B
# → mode: ternary
```

Uses Straight-Through Estimator (STE) for gradient flow through the quantization function.

### Pure PTQ

No teacher needed — post-training quantization with per-channel symmetric scaling.

---

## Hardware Requirements

| Setup | Best For |
|-------|----------|
| **2x T4 (15.9GB each) + 32GB RAM** | MoE models up to 30B params |
| **1x T4/RTX 3090/4090** | Dense models up to 8B |
| **CPU only** | Small models (<3B) |

Unsloth handles automatic GPU↔CPU offloading. Expert batching keeps peak VRAM under 2GB regardless of model size.

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
1. Student model (any HF model name or local path)
2. Teacher model (optional — enables distillation)
3. Quantization mode (1-bit or ternary)
4. Expert batch size (auto or manual)
5. Output directory

---

## Usage

### Interactive Mode (recommended)

```bash
./quant.sh
```

### Step-by-Step Examples

#### 1. Ternary quantize a small dense model (fits on any GPU)

```bash
git clone https://github.com/Abdulhadi446/quant-it.git
cd quant-it
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

#### 3. Ternary quantize a MoE model on 2x T4

```bash
./quant.sh
# → student: Qwen/Qwen3-30B-A3B
# → teacher: (empty — PTQ)
# → mode: ternary
# → batch: auto (fits ~50 experts per batch on 2x T4)
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

#### 5. Ternary quantize Mistral-7B

```bash
./quant.sh
# → student: mistralai/Mistral-7B-v0.1
# → teacher: (empty)
# → mode: ternary
# → output: Mistral-7B-v0.1-ternary
```

#### 6. 1-bit quantize Phi-3 with custom calibration data

```bash
# first create a calibration file
echo "The capital of France is Paris." > calib.txt
echo "Machine learning is a subset of AI." >> calib.txt
echo "Python is a popular programming language." >> calib.txt

./quant.sh
# → student: microsoft/Phi-3-mini-4k-instruct
# → teacher: (empty)
# → mode: 1bit
# → output: Phi-3-mini-4k-instruct-1bit
```

#### 7. Quantize a local model

```bash
./quant.sh
# → student: /path/to/my/local/model
# → teacher: (empty)
# → mode: ternary
# → output: my-local-model-ternary
```

#### 8. Full distillation with custom hyperparameters

```bash
./quant.sh
# → student: Qwen/Qwen2.5-7B
# → teacher: Qwen/Qwen2.5-72B
# → mode: ternary
# → epochs: 5
# → lr: 1e-4
# → batch size: 2
# → max len: 1024
# → calib file: calib.txt
# → output: Qwen2.5-7B-ternary-distilled
```

### Programmatic Usage

```python
from quantize import ptq, ternary_weight, onebit_weight
from transformers import AutoModelForCausalLM

# ternary quantization
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = ptq(model, ternary_weight)
model.save_pretrained("qwen2.5-0.5b-ternary")

# 1-bit quantization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = ptq(model, onebit_weight)
model.save_pretrained("llama3-8b-1bit")
```

### Running from Jupyter / IPython

Shell commands need `!` prefix in Jupyter:

```python
# wrong — causes SyntaxError
# ./quant.sh

# correct
!./quant.sh

# or run the Python script directly
!python quantize.py
```

### One-Liner Examples

```bash
# clone + install + run in one shot
git clone https://github.com/Abdulhadi446/quant-it.git && cd quant-it && ./quant.sh

# re-run after initial install (skips venv creation)
source .venv/bin/activate && python quantize.py

# quantize and push to HuggingFace
python quantize.py Qwen/Qwen2.5-0.5B -o qwen-ternary && \
  huggingface-cli upload my-username/qwen-2.5-0.5b-ternary qwen-ternary
```

---

## How It Works

### Quantization Functions

**1-bit (Binary)**: Each weight is replaced by its sign — `{-1, +1}`.

```python
w_binary = sign(w)  # per-output-channel
```

**Ternary**: Symmetric ternary with learned thresholds via `tanh` normalization:

```python
scale = mean(|w|)           # per-output-channel scaling
w_norm = w / scale
w_ternary = tanh(w_norm)    # squash to [-1, +1]
# threshold at ±0.4 → {-1, 0, +1}
```

### MoE Router Detection

The script scans model architecture for common MoE patterns:
- `gate_proj`, `router`, `router_proj`
- `mlp.gate`, `block_sparse_moe.gate`
- `expert` + `nn.Linear` layers

Router weights are preserved in their original precision to maintain routing accuracy.

### Expert Batch Sizing

```
available VRAM × safety_factor / expert_size = batch_size
```

With 2x T4 (~28GB free) and 300MB experts: `28 × 0.6 / 0.3 = 56 experts per batch`

---

## Quantization Research References

This tool implements techniques from:

- **[Bonsai](https://github.com/deepgrove-ai/Bonsai)** — 500M ternary-weight LM trained in <5B tokens (deepgrove-ai)
- **[PrismML Bonsai](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit and ternary models from 1.7B to 27B parameters
- **[Bonsai 27B Whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)** — 14x less memory, 8x faster, 5x less energy vs FP16
- **[1-bit Bonsai 8B](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)** — Q1_0 group-128 binary quantization
- **[Ternary Bonsai 8B](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)** — Q2_0 ternary quantization with g128 grouping
- **[Bonsai Image 4B](https://github.com/PrismML-Eng/Bonsai-Image-Demo/blob/main/bonsai-image-4b-whitepaper.pdf)** — Vision-language ternary model

### Key Technical Insights Applied

| Concept | Source | Implementation |
|---------|--------|----------------|
| Per-channel symmetric scaling | Bonsai papers | `scale = mean(|w|)` per output dim |
| Tanh-based ternary thresholding | PrismML | threshold at ±0.4 after tanh normalization |
| MoE router preservation | Bonsai MoE analysis | skip `gate_proj`, `router` layers |
| Group-128 quantization | Q1_0/Q2_0 GGUF format | group size 128 for weight blocks |
| Expert streaming | Hardware constraints | batch-load experts, quantize, offload |
| STE for distillation | Standard QAT | straight-through estimator for gradient flow |

---

## Model Compatibility

Works with any HuggingFace `AutoModelForCausalLM`:

- **Dense**: Llama, Qwen, Mistral, Phi, Gemma, GPT-2, etc.
- **MoE**: Qwen-MoE, Mixtral, DeepSeek-V2/V3, DBRX, etc.
- **Custom**: any model with `trust_remote_code=True`

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ system RAM

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

## Output

Quantized models are saved in HuggingFace format:

```
model_name-ternary/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

Load and use like any HuggingFace model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("llama3-8b-ternary")
tokenizer = AutoTokenizer.from_pretrained("llama3-8b-ternary")
```

---

## Benchmarks (Bonsai Reference)

The Bonsai family demonstrates what extreme quantization achieves:

| Model | Params | Format | Weights Size | MMLU | ARC-c |
|-------|--------|--------|-------------|------|-------|
| Bonsai (ternary) | 500M | Q2_0 | ~170 MB | 30.28 | 33.36 |
| Qwen 2.5 0.5B (FP16) | 500M | FP16 | 1.0 GB | 33.40 | 32.25 |
| Bonsai-8B (1-bit) | 8B | Q1_0 | ~1.0 GB | — | — |
| Bonsai-27B (1-bit) | 27B | Q1_0 | 3.53 GiB | — | — |
| Bonsai-27B (ternary) | 27B | Q2_0 | 6.66 GiB | — | — |

Ternary Bonsai-500M matches FP16 models 10x its size on common benchmarks.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [deepgrove-ai](https://github.com/deepgrove-ai/Bonsai) for the original Bonsai ternary model
- [PrismML](https://github.com/PrismML-Eng/Bonsai-demo) for the Bonsai 1-bit and ternary family
- [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient model loading and training
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for Q1_0 and Q2_0 quantization format support
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the model ecosystem
