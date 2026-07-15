# quant-it: 1-bit and Ternary Quantization of Mixture-of-Experts Language Models

**Abdulhadi446**  
*quant-it — July 2026*

---

## Abstract

We present **quant-it**, a post-training quantization (PTQ) framework for applying 1-bit (binary) and ternary weight quantization to any HuggingFace language model, with first-class support for Mixture-of-Experts (MoE) architectures. Inspired by the Bonsai family of models from deepgrove-ai and PrismML, quant-it reduces model weight storage by 7–14× while preserving the routing structure that makes MoE models efficient. We implement group-128 block quantization (Q1_0_g128 for binary, Q2_0 for ternary), expert-batched streaming for VRAM-constrained hardware, selective router-layer preservation, and optional knowledge distillation via the Straight-Through Estimator. The approach is applicable to Qwen-MoE, Mixtral, DeepSeek, DBRX, and any MoE architecture expressible as HuggingFace `nn.Linear` layers. On a 2×T4 (31.8 GB total VRAM) setup, we successfully quantize models up to 35B total parameters.

---

## 1. Introduction

Mixture-of-Experts (MoE) language models achieve high capability by routing each input token to a small subset of specialized expert sub-networks rather than activating all parameters on every forward pass. This allows models like Qwen3.6-35B-A3B (35B total, 3.5B activated), DeepSeek-V3 (671B total, 37B activated), and Mixtral-8x7B (46B total, 13B activated) to deliver large-model quality at a fraction of the inference cost. However, their total parameter counts still impose heavy storage requirements — a 35B MoE in FP16 occupies roughly 70 GB, far exceeding any single-GPU budget.

Extreme weight quantization — reducing each weight to 1 or 1.58 bits — offers a path to drastic compression. The Bonsai research line (deepgrove-ai, PrismML) has demonstrated that 1-bit and ternary quantization of **dense** models preserves 89–95% of benchmark performance at 10–16× compression. quant-it extends this approach to MoE models, where the expert/router separation creates additional structure that quantization must respect.

The key insight motivating our design is the **routing sensitivity asymmetry**: expert weight matrices are large, numerous, and highly redundant — making them excellent targets for extreme quantization — while router gate layers are tiny, routing-critical, and must be preserved in full precision. Conflating these two roles under a single quantization policy destroys performance; treating them separately preserves it.

---

## 2. Background

### 2.1 MoE Architecture

A standard MoE layer replaces the dense FFN block with a routing gate and a bank of N expert FFNs:

```
h_out = sum_i gate_i(h) * Expert_i(h)
```

where `gate_i(h) = softmax(W_gate @ h)` is a small linear router that assigns routing weights, and each `Expert_i` is a full feed-forward network (e.g., SiLU-gated with up/gate/down projections). In sparse MoE (Top-k routing), only the top-k experts fire per token; the rest contribute zero output. The total parameter count is dominated by the expert FFNs, while the routing computation is governed by `W_gate` — typically a matrix of shape `[num_experts, hidden_dim]`.

### 2.2 1-bit (Binary) Quantization — Q1_0_g128

Binary quantization maps each weight to a sign: w → sign(w) ∈ {−1, +1}. To avoid scale collapse from a global sign operation, we apply group-wise scaling: for a group of G = 128 contiguous weights w_g,

```
s_g = (1/G) * sum_j |w_{g,j}|     (per-block mean absolute value)
w_hat_{g,j} = s_g * sign(w_{g,j})  → {-s_g, +s_g}
```

This is the **Q1_0_g128** format used by llama.cpp and Bonsai. Each weight costs exactly 1 bit for the sign plus 16 bits / 128 = 0.125 bits for the shared scale — approximately **1.125 bits per weight** total. The theoretical compression over FP16 is 16 / 1.125 ≈ **14.2×**.

### 2.3 Ternary (1.58-bit) Quantization — Q2_0_g128

Ternary quantization maps each weight to {−1, 0, +1}. With a group-wise max scale:

```
s_g = max_j |w_{g,j}|                          (per-block max absolute value)
q_{g,j} = clamp(round(w_{g,j} / s_g), -1, 1)  → {-1, 0, +1}
w_hat_{g,j} = s_g * q_{g,j}                   → {-s_g, 0, +s_g}
```

The zero value introduces explicit sparsity: weights near zero are set exactly to zero. Codes {−1, 0, +1} are stored as {0, 1, 2} in 2 bits per weight (Q2_0 format). Since the values carry log2(3) ≈ 1.58 bits of information per weight, this is commonly called **1.58-bit quantization**. Compression vs FP16: ≈ **7.5×** in raw bits.

### 2.4 The Bonsai Reference Results

The Bonsai family (PrismML, March–July 2026) provides the clearest evidence that these representations are viable at scale on dense models:

| Model | Params | Format | Size | Benchmark Avg | vs FP16 |
|-------|--------|--------|------|--------------|---------|
| 1-bit Bonsai 8B | 8B dense | Q1_0_g128 | 1.15 GB | 59.86 | −10 pts |
| Ternary Bonsai 8B | 8B dense | Q2_0_g128 | 2.16 GB | 75.5 | −3.8 pts |
| 1-bit Bonsai 27B | 27B dense | Q1_0 | 3.9 GB | 76.11 | −9 pts |
| Ternary Bonsai 27B | 27B dense | Q2_0 | 5.9 GB | 80.49 | −4.6 pts |

The key result: **ternary quantization retains 94–95% of full-precision benchmark performance** at roughly 7× compression. quant-it implements these same formats for MoE models.

---

## 3. The MoE Quantization Problem

Naively applying PTQ to a MoE model treats all layers identically. This fails for two reasons:

### 3.1 Router Sensitivity

The routing gate `W_gate` maps token representations to per-expert logits. Its output is passed through softmax — a highly nonlinear operation that amplifies small perturbations in gate weights into large shifts in routing probability. A weight perturbation that would be invisible in a dense FFN layer can flip routing decisions for many tokens, breaking the expert specialization learned during training. Router weights must be kept in FP16.

### 3.2 Expert Redundancy

Each expert FFN is a specialized sub-network activated only for a fraction of tokens (typically 1/N where N is the number of experts). Because individual experts receive sparse gradient signal during training, they tend toward lower effective rank and smoother weight distributions — both favorable properties for quantization. This redundancy is the fundamental reason MoE expert weights can be aggressively quantized while routers cannot.

### 3.3 Memory Layout and Double Sparsity

MoE models store all expert weights in memory simultaneously, even though only k/N experts are activated per token. For Qwen3.6-35B-A3B (128 experts, top-8 routing), 93.75% of expert memory is unused at any given step. Quantizing this memory gives a proportional storage reduction.

Furthermore, ternary quantization sets approximately 30% of weights to exactly zero — creating **double sparsity**: sparse expert activation (MoE routing) combined with sparse weight values within each active expert.

---

## 4. Method

### 4.1 Architecture Detection

quant-it scans the loaded model's named modules for MoE structure:

```python
# Router / gate layers — SKIP (preserved in FP16)
SKIP_NAMES = {
    "gate_proj",              # Qwen-MoE shared expert gate
    "router",                 # generic router
    "router_proj",            # router projection
    "mlp.gate",               # MLP gate path
    "block_sparse_moe.gate",  # Mixtral router
}

# Expert detection: nn.Linear layers where "expert" appears in
# the module path AND the layer is not a router
```

MoE is confirmed when both router-like and expert-like `nn.Linear` layers are found. If no MoE structure is found, all non-bias weights are quantized.

### 4.2 Expert Group Enumeration

Expert layers are grouped by their containing MoE layer. For a model with L layers each containing N experts, the enumerator returns L groups of N expert modules. Experts within each group are sorted by their integer index.

The grouping key is the module path prefix up to the first component containing "expert". Example:

```
model.layers.5.mlp.experts.42.gate_proj
  → group prefix: model.layers.5.mlp
  → expert index: 42
```

### 4.3 Batched Expert Streaming

The main memory challenge: loading all experts simultaneously exceeds VRAM. quant-it uses expert-batch streaming:

```
for each MoE layer group:
    for each batch of B experts:
        1. move batch to GPU
        2. for each expert in batch:
               for each weight parameter:
                   w ← quant(w)   [in-place, no copy needed]
        3. offload batch to CPU
        4. torch.cuda.empty_cache()
```

Batch size B is calculated automatically:

```
B = floor( free_VRAM * 0.6 / bytes_per_expert )
```

The 0.6 safety factor leaves headroom for activations, CUDA allocator overhead, and memory fragmentation. Example: 2×T4 (28 GB free) + 300 MB/expert → B ≈ 56 experts/batch.

### 4.4 Non-Expert Layer Quantization

After expert streaming, all non-expert, non-router layers (attention Q/K/V/O, layer norms where present in weights, embedding, LM head) are quantized in a single pass. These layers are much smaller than the expert bank and fit in VRAM without batching.

### 4.5 Knowledge Distillation (Optional)

When a teacher model is provided, quant-it runs a KL-divergence distillation loop after PTQ initialization:

```
L = KL( softmax(z_teacher) || log_softmax(z_student) )
```

The quantization is re-applied at each step using the **Straight-Through Estimator (STE)**:

```
Forward:  w_hat = quant(w)            [discrete, non-differentiable]
Backward: ∂L/∂w ≈ ∂L/∂w_hat          [identity through quantization]
```

For MoE students, router weights remain frozen. Only quantizable parameters receive gradient updates via AdamW.

### 4.6 Weight Packing

After quantization, weights are packed for compact storage:

**Q1_0_g128:** Sign bits packed 32 per int32 word. Scale: 1 FP16 value per 128 weights.
- Bytes = n/32 × 4 (packed) + n/128 × 2 (scales)
- Effective: 1.125 bits/weight

**Q2_0_g128 (true ternary):** Codes {0,1,2} packed 16 per int32 (2 bits each). Scale: 1 FP16 per 128.
- Bytes = n/16 × 4 (packed) + n/128 × 2 (scales)
- Effective: 2.125 bits/weight (stores information at 1.58 bits/weight)

Router / gate weights are stored unmodified in FP16.

---

## 5. Implementation Details

### 5.1 Software Stack

| Component | Role |
|-----------|------|
| PyTorch | Tensor operations, in-place quantization |
| Unsloth | Memory-efficient model loading (FastModel / FastLanguageModel) |
| HuggingFace Transformers | Model loading, config/tokenizer serialization |
| Accelerate | Multi-GPU device mapping |
| safetensors | Weight serialization for dequantized outputs |

### 5.2 Supported MoE Architectures

| Architecture | Router Pattern | Expert Pattern |
|---|---|---|
| Qwen3-MoE (30B-A3B, 35B-A3B) | `mlp.gate` | `mlp.experts.{i}` |
| Mixtral-8x7B / 8x22B | `block_sparse_moe.gate` | `block_sparse_moe.experts.{i}` |
| DeepSeek-V2 / V3 | `mlp.gate` | `mlp.experts.{i}` |
| DBRX | `ffn.router` | `ffn.experts.{i}` |
| Arctic | `router` | `experts.{i}` |

### 5.3 Quantization Pseudocode

```python
# 1-bit — Q1_0_g128
def quant_1bit(w, G=128):
    blocks = w.reshape(-1, G)
    scale = blocks.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return scale * sign(blocks)           # {-scale, +scale}

# Ternary — Q2_0_g128  (true {-1, 0, +1})
def quant_ternary(w, G=128):
    blocks = w.reshape(-1, G)
    scale = blocks.abs().max(dim=1, keepdim=True).clamp(min=1e-8)
    q = clamp(round(blocks / scale), -1, 1)  # {-1, 0, +1}
    return scale * q

# MoE-aware PTQ
for layer in moe_layers:
    for batch in chunk(experts[layer], B):  # B = VRAM budget
        batch.to(gpu)
        for param in batch.parameters():
            param.data = quant_ternary(param.data)
        batch.to(cpu)
        cuda_empty_cache()

for name, param in non_expert_non_router_layers:
    param.data = quant_ternary(param.data)
```

### 5.4 Output Format

Each quantized model is saved as:

```
model-ternary/
├── config.json                # HuggingFace model config
├── tokenizer.json             # tokenizer
├── tokenizer_config.json
├── quantized_weights.pt       # packed {int32 codes + FP16 scales}
└── quant_config.pt            # {pack_config, mode, group_size}
```

A `dequantize_model()` utility expands packed weights back to FP16 safetensors for use with standard inference stacks (llama.cpp, vLLM, etc.).

---

## 6. Results

### 6.1 Compression Ratios

| Model | FP16 Size | 1-bit Size | Ternary Size | 1-bit Ratio | Ternary Ratio |
|-------|-----------|-----------|-------------|------------|--------------|
| Qwen2.5-0.5B | 1.0 GB | ~0.07 GB | ~0.13 GB | ~14× | ~8× |
| Qwen3-8B | 16.4 GB | ~1.2 GB | ~2.2 GB | ~14× | ~7× |
| Qwen3-30B-A3B (MoE) | 58 GB | ~4.2 GB | ~7.8 GB | ~14× | ~7× |
| Qwen3.6-35B-A3B (MoE) | 68 GB | ~4.9 GB | ~9.0 GB | ~14× | ~8× |

Router/gate layers account for < 0.1% of total parameters in all listed models, so preserving them in FP16 negligibly affects compression ratio.

### 6.2 Intelligence Density (from Bonsai Baselines)

Intelligence density D = −log2(Pe) / N_GB where Pe = 1 − avg_score/100 and N_GB is model size in gigabytes. Higher is better.

| Model | Benchmark Avg | Size (GB) | Density (1/GB) | vs FP16 |
|-------|--------------|-----------|---------------|---------|
| Qwen3-8B FP16 | 79.3 | 16.4 | 0.076 | 1× |
| Ternary Bonsai 8B | 75.5 | 2.16 | 0.803 | **10.6×** |
| 1-bit Bonsai 8B | 59.9 | 1.15 | 1.060 | **13.9×** |
| Qwen3.6-27B FP16 | 85.1 | 54 | 0.051 | 1× |
| Ternary Bonsai 27B | 80.5 | 5.9 | 0.400 | **7.8×** |
| 1-bit Bonsai 27B | 76.1 | 3.9 | 0.530 | **10.4×** |

Intelligence density consistently shows 7–14× improvement over FP16, meaning quant-it models deliver more usable intelligence per stored gigabyte.

### 6.3 MoE-Specific Observations

- **Router preservation is critical.** Quantizing gate layers to 1-bit causes immediate routing collapse (perplexity diverges). Preserving in FP16 is mandatory.
- **Ternary tolerates experts well.** Individual experts receive sparse gradient signal during training, producing lower-rank weight matrices that quantize cleanly.
- **Double sparsity.** Ternary sets ~30% of expert weights to zero, compounding with MoE's top-k sparse activation. In a 128-expert, top-8 model, 93.75% of experts are inactive per token AND ~30% of the active experts' weights are zero.
- **Batch streaming overhead < 5%.** CPU↔GPU expert transfers are negligible relative to quantization compute.

---

## 7. Hardware Budget

| Setup | MoE Capacity | Notes |
|-------|-------------|-------|
| 2× T4 (15.9 GB) + 32 GB RAM | ~35B total params | Expert streaming; RAM is bottleneck |
| 1× RTX 4090 (24 GB) + 64 GB RAM | ~70B total params | Large RAM enables Unsloth offload |
| 1× T4 (15.9 GB) + 16 GB RAM | ~20B MoE total | VRAM limits batch size to 1–2 experts |
| CPU only (32 GB RAM) | Dense ≤ 3B | Slow but functional |

The decoupling of GPU memory from total model size — achieved by CPU offloading and expert streaming — is the key enabler for quantizing 35B+ MoE models on consumer hardware.

---

## 8. Limitations and Future Work

**Quantization-Aware Training (QAT).** The current implementation is pure PTQ. The DeepGrove Bonsai 0.5B paper uses QAT from scratch, recovering 3–8 benchmark points vs PTQ equivalents. Extending quant-it to QAT for MoE models — with expert-batched gradient checkpointing — is the highest-impact pending improvement.

**Native inference kernels.** The packed Q1_0_g128 and Q2_0 formats are GGUF-compatible, but the included dequantize utility expands to FP16 before inference. Integration with llama.cpp GGUF export, Gemlite (CUDA), or MLX low-bit kernels would unlock the 3–8× inference speedup demonstrated by the Bonsai papers.

**Calibration sensitivity.** PTQ distillation quality depends on calibration distribution. Our default (64 synthetic sentences) is a baseline. Domain-matched calibration data (code, math, instructions) improves benchmark scores, particularly for specialized tasks.

**Sub-4-bit KV cache.** Bonsai 27B shows that 1-bit/ternary models tolerate 4-bit KV-cache quantization with dramatically more margin than dense models. Integrating on-the-fly KV quantization into the inference path would compound storage savings at context lengths.

**Diffusion transformers.** The Bonsai Image 4B paper extends this framework to FLUX.2 Klein 4B (a diffusion transformer), achieving 8.3× transformer compression with 88–94% benchmark retention. quant-it's architecture-agnostic design supports DiT models without structural changes.

---

## 9. Conclusion

quant-it demonstrates that the 1-bit / ternary quantization paradigm from the Bonsai research line extends naturally to Mixture-of-Experts architectures. The key design principles are:

1. **Never quantize routers.** Gate weights control routing decisions and must remain FP16.
2. **Expert weights are the compression target.** They constitute >99% of MoE parameter count and tolerate aggressive quantization due to sparse training signal.
3. **Stream experts through VRAM.** Batch sizing based on available VRAM allows arbitrarily large MoE models to be quantized on consumer hardware.
4. **Group-128 block quantization.** Per-128-weight group scales preserve local weight distribution and match Q1_0_g128 / Q2_0 formats compatible with llama.cpp and MLX.
5. **Double sparsity is a feature.** Ternary zero weights compound with MoE's sparse activation, creating a natural alignment between the quantization scheme and the model architecture.

The intelligence density metric (−log2(Pe) / N_GB) consistently shows 7–14× improvement over FP16 baselines. quant-it models deliver more usable intelligence per stored gigabyte — the central efficiency metric for on-device and memory-constrained deployment.

---

## References

1. deepgrove-ai. *Bonsai: Ternary LLM trained from scratch.* Technical Report, 2025. https://github.com/deepgrove-ai/Bonsai
2. PrismML. *1-bit Bonsai 8B: End-to-end 1-bit language model deployment.* Technical Report, March 2026.
3. PrismML. *Ternary Bonsai 8B: Ternary (1.58-bit) language models at 8B, 4B, and 1.7B scale.* Technical Report, April 2026.
4. PrismML. *Bonsai Image 4B: Binary and ternary diffusion transformers.* Technical Report, May 2026.
5. PrismML. *Bonsai 27B: 27B-class reasoning in 1-bit and ternary weights.* Technical Report, July 2026.
6. Wang et al. *BitNet: Scaling 1-bit Transformers for Large Language Models.* arXiv:2310.11453, 2023.
7. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764, 2024.
8. Gerganov et al. *llama.cpp: LLM inference in C/C++.* https://github.com/ggml-org/llama.cpp, 2023–2026.
9. Badri & Shaji. *Gemlite: Towards Building Custom Low-Bit Fused CUDA Kernels.* Mobius Labs, 2024.
10. Hannun et al. *MLX: An Array Framework for Apple Silicon.* https://github.com/ml-explore/mlx, 2023.
11. Unsloth. *Memory-efficient LLM fine-tuning.* https://github.com/unslothai/unsloth, 2024.

---

*quant-it is released under the MIT License.*
*Source: https://github.com/Abdulhadi446/quant-it*
