#!/usr/bin/env python3
"""
Interactive universal 1-bit / ternary quantizer using Unsloth.
Optimized for 2x T4 (15.9GB each) + 32GB CPU RAM.
Auto-detects MoE, skips router layers, batched expert-by-expert streaming.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# set HF cache to persistent directory
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    # Kaggle: /tmp has more disk space than /kaggle/working
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/hf_cache"
elif "HF_HOME" not in os.environ:
    # local: use ~/.cache (not ./cache)
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# quantization primitives — block-based (Bonsai Q1_0 / Q2_0 style)
# ---------------------------------------------------------------------------

def quant_q1_0(tensor, group_size=128):
    """Q1_0_g128: 1-bit block quantization with group size 128 (matches Bonsai Q1_0_g128).
       scale = max(|w|) per block (absmax), sign encodes the bit.
       weights -> {-scale, +scale} within each group.
       In-place. Returns quantized tensor.
    """
    w = tensor.float()
    orig_shape = w.shape
    flat = w.view(-1)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, group_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    bits = (blocks >= 0).to(flat.dtype) * 2 - 1  # {0,1} -> {-1,+1}
    quantized = bits * scales
    if pad:
        quantized = quantized.view(-1)[:n]
    tensor.data.copy_(quantized.view(orig_shape).to(tensor.dtype))
    return tensor


def quant_q2_0(tensor, group_size=128):
    """True ternary block quantization: weights -> {-1, 0, +1} * scale.
       scale = max(|w|) per block (g128, matches Bonsai Q2_0 / 1.58-bit).
       q_raw = round(w / scale) clamped to {-1, 0, +1}.
       stored code = q_raw + 1  -> {0, 1, 2}  (2 bits per weight).
       decode: (code - 1) * scale -> {-scale, 0, +scale}.
       In-place. Returns quantized tensor.
    """
    w = tensor.float()
    orig_shape = w.shape
    flat = w.view(-1)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, group_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    q = (blocks / scales).round().to(torch.int32).clamp(-1, 1)  # {-1, 0, +1}
    decoded = q.float() * scales
    if pad:
        decoded = decoded.view(-1)[:n]
    tensor.data.copy_(decoded.view(orig_shape).to(tensor.dtype))
    return tensor


# ---------------------------------------------------------------------------
# bit-packing (stores scales + packed bits for real compression)
# ---------------------------------------------------------------------------

def pack_q1_0_blocks(tensor, scales, group_size=128):
    """Pack Q1_0_g128 quantized weights + scales into compact representation.
       32 sign bits per int32 word.  Returns: dict with packed, scales, n, group_size.
    """
    w = tensor.float()
    flat = (w.view(-1) >= 0).to(torch.int32)  # int32 to avoid uint8 overflow on shifts
    n = flat.numel()
    pad = (32 - n % 32) % 32
    if pad:
        flat = F.pad(flat, (0, pad))
    bits_blocks = flat.view(-1, 32)
    powers = (2 ** torch.arange(32, dtype=torch.int64)).to(torch.int32)  # int32 powers
    packed = (bits_blocks * powers).sum(dim=1).to(torch.int32)
    return {
        "packed": packed.cpu(),
        "scales": scales.cpu().half(),
        "n": n,
        "group_size": group_size,
    }


def pack_q2_0_blocks(q_codes, scales, group_size=128):
    """Pack ternary codes {0,1,2} (= q_raw+1) + scales into compact representation.
       Each code = 2 bits, 16 codes per int32.
       Returns: dict with packed, scales, n, group_size.
    """
    q = q_codes.to(torch.int32).view(-1)  # {0,1,2}
    n = q.numel()
    pad = (16 - n % 16) % 16
    if pad:
        q = F.pad(q, (0, pad))
    code_blocks = q.view(-1, 16)
    shifts = 2 * torch.arange(16, dtype=torch.int32)
    packed = (code_blocks << shifts).sum(dim=1)
    return {
        "packed": packed.cpu(),
        "scales": scales.cpu().half(),
        "n": n,
        "group_size": group_size,
    }


def unpack_q1_0(packed_data):
    """Unpack Q1_0 back to FP32 tensor."""
    packed = packed_data["packed"].to(torch.int32)
    scales = packed_data["scales"].float()
    n = packed_data["n"]
    group_size = packed_data["group_size"]
    bits = ((packed.view(-1, 1) >> torch.arange(32, device=packed.device, dtype=torch.int32)) & 1).to(torch.float32)
    bits = bits.view(-1)[:n] * 2 - 1
    n_blocks = (n + group_size - 1) // group_size
    scales_repeated = scales[:n_blocks].repeat_interleave(group_size)[:n]
    return bits * scales_repeated


# Bonsai Q2_0 ternary decode table: 2-bit code -> weight multiplier
# 0b00(0)->-1, 0b01(1)->0, 0b10(2)->+1, 0b11(3)->0 (duplicate zero)
DECODE_Q2 = torch.tensor([-1.0, 0.0, 1.0, 0.0])


def unpack_q2_0(packed_data):
    """Unpack ternary Q2_0 back to FP32 tensor."""
    packed = packed_data["packed"].to(torch.int32)
    scales = packed_data["scales"].float()
    n = packed_data["n"]
    group_size = packed_data["group_size"]
    vals = ((packed.view(-1, 1) >> (2 * torch.arange(16, device=packed.device, dtype=torch.int32))) & 3)
    codes = vals.view(-1)[:n].clamp(0, 3).to(torch.long)  # Bonsai: code3 maps to 0
    n_blocks = (n + group_size - 1) // group_size
    scales_repeated = scales[:n_blocks].repeat_interleave(group_size)[:n]
    return DECODE_Q2.to(codes.device)[codes].float() * scales_repeated


# wrappers for ptq() / distill() — call in-place quant then return tensor
def onebit_weight(w):
    """1-bit binary quantization with g128 grouping (Bonsai Q1_0_g128)."""
    return quant_q1_0(w, group_size=128)


def ternary_weight(w):
    """True ternary quantization with g128 grouping (Bonsai Q2_0 / 1.58-bit)."""
    return quant_q2_0(w, group_size=128)


QUANT = {"1": ("1bit", onebit_weight), "2": ("ternary", ternary_weight)}


def dequantize_model(packed_dir, output_dir, dtype=torch.float16):
    """Unpack quantized weights back to FP16 for inference."""
    from transformers import AutoConfig
    from safetensors.torch import save_file as st_save
    from transformers import AutoTokenizer

    packed_dir = Path(packed_dir)
    config = AutoConfig.from_pretrained(packed_dir)
    state = torch.load(packed_dir / "quantized_weights.pt", map_location="cpu", weights_only=True)
    config_data = torch.load(packed_dir / "quant_config.pt", map_location="cpu", weights_only=True)
    pack_config = config_data["pack_config"]

    print(f"dequantizing {len(state)} layers...")
    for name, packed_data in state.items():
        if name not in pack_config:
            continue
        fmt = pack_config[name]
        if fmt == "fp16":
            continue
        if fmt == "1bit":
            state[name] = unpack_q1_0(packed_data).to(dtype)
        elif fmt == "ternary":
            state[name] = unpack_q2_0(packed_data).to(dtype)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    if (packed_dir / "tokenizer.json").exists():
        AutoTokenizer.from_pretrained(packed_dir).save_pretrained(output_dir)

    st_state = {}
    for name, tensor in state.items():
        st_name = name.replace("model.", "", 1) if name.startswith("model.") else name
        st_state[st_name] = tensor.contiguous()

    st_save(st_state, output_dir / "model.safetensors")
    print(f"dequantized → {output_dir}  ({len(state)} layers)")


# ---------------------------------------------------------------------------
# GGUF export (for llama.cpp / ollama / lmstudio)
# ---------------------------------------------------------------------------

# HF → GGUF tensor name mapping for common architectures
_TENSOR_MAP = {
    "embed_tokens": "token_embd",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "mlp.experts": "ffn_gate",  # MoE experts → handled separately
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "linear_attn.norm": "linear_norm",
    "self_attn.q_norm": "attn_norm_q",
    "self_attn.k_norm": "attn_norm_k",
    "self_attn.norm": "attn_norm",
}


def _map_tensor_name(name, arch):
    """Map HF tensor name to GGUF name."""
    parts = name.split(".")
    # embedding
    if "embed_tokens" in name:
        return "token_embd.weight"
    # final norm (model.norm.weight)
    if name.endswith(".norm.weight") and "layers" not in name:
        return "output_norm.weight"
    # lm_head
    if "lm_head" in name:
        return "output.weight"
    # layer tensors
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer_idx = parts[i + 1]
            rest = ".".join(parts[i + 2:])
            # MoE expert handling
            if "experts" in rest:
                exp_match = rest.split("experts.")
                if len(exp_match) > 1:
                    exp_idx = exp_match[1].split(".")[0]
                    expert_part = ".".join(exp_match[1].split(".")[1:])
                    for ehf, egguf in _TENSOR_MAP.items():
                        if ehf in expert_part:
                            return f"blk.{layer_idx}.{egguf}.{exp_idx}.weight"
            # Standard tensor mapping
            for hf_key, gguf_key in _TENSOR_MAP.items():
                if hf_key in rest:
                    suffix = rest.split(hf_key)[-1]
                    return f"blk.{layer_idx}.{gguf_key}{suffix}"
    return name


def save_gguf(packed_dir, output_path):
    """Convert quantized model to GGUF format (FP16).
       Unpacks packed weights incrementally to avoid OOM.
    """
    try:
        import gguf
    except ImportError:
        print("  installing gguf package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf", "-q"])
        import gguf

    from transformers import AutoConfig

    packed_dir = Path(packed_dir)
    config = AutoConfig.from_pretrained(packed_dir)
    state = torch.load(packed_dir / "quantized_weights.pt", map_location="cpu", weights_only=True)
    config_data = torch.load(packed_dir / "quant_config.pt", map_location="cpu", weights_only=True)
    pack_config = config_data["pack_config"]

    arch = getattr(config, "model_type", "llama")
    n_layers = getattr(config, "num_hidden_layers", 32)
    n_heads = getattr(config, "num_attention_heads", 32)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    hidden = getattr(config, "hidden_size", 4096)
    intermediate = getattr(config, "intermediate_size", 11008)
    vocab = getattr(config, "vocab_size", 32000)
    max_ctx = getattr(config, "max_position_embeddings", 4096)
    rms_eps = getattr(config, "rms_norm_eps", 1e-6)
    rope_theta = getattr(config, "rope_theta", 10000.0)

    print(f"writing GGUF → {output_path}")
    print(f"  arch: {arch}  layers: {n_layers}  hidden: {hidden}")

    writer = gguf.GGUFWriter(str(output_path), arch)
    writer.add_name(packed_dir.name)
    writer.add_context_length(max_ctx)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_block_count(n_layers)
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(rms_eps)
    writer.add_file_type(0)  # ALL_F32 = 0

    # write tensors — unpack from packed format, write as F16
    for name, packed_data in state.items():
        fmt = pack_config.get(name, "fp16")
        if fmt == "fp16":
            tensor = packed_data.to(torch.float16)
        elif fmt == "1bit":
            tensor = unpack_q1_0(packed_data).to(torch.float16)
        elif fmt == "ternary":
            tensor = unpack_q2_0(packed_data).to(torch.float16)
        else:
            tensor = packed_data.to(torch.float16)

        gguf_name = _map_tensor_name(name, arch)
        writer.add_tensor(gguf_name, tensor.numpy())
        print(f"  {name} → {gguf_name}  {list(tensor.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(output_path).stat().st_size / (1024**2)
    orig_mb = sum(p.numel() for p in state.values()) * 2 / (1024**2)
    print(f"\n  GGUF saved → {output_path}")
    print(f"  size: {size_mb:.0f} MB  (FP16, original was {orig_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# MoE detection + expert enumeration
# ---------------------------------------------------------------------------

SKIP_NAMES = {"gate_proj", "router", "router_proj", "mlp.gate", "block_sparse_moe.gate"}


def is_moe(model):
    has_gate = False
    has_expert = False
    for name, mod in model.named_modules():
        nl = name.lower()
        if any(k in nl for k in ("gate_proj", "router", "mlp.gate", "block_sparse_moe")):
            if isinstance(mod, nn.Linear):
                has_gate = True
        if "expert" in nl:
            has_expert = True
    return has_gate and has_expert


def should_skip(name):
    return any(s in name.lower() for s in SKIP_NAMES)


def find_expert_groups(model):
    """
    Find all MoE expert layers and group them.
    Returns list of dicts:
      {layer_path, expert_indices, expert_modules: [(idx, module), ...]}
    """
    # find all linear layers with "expert" in the name
    expert_layers = {}  # layer_prefix -> [(idx, full_name, module)]
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        nl = name.lower()
        if "expert" not in nl:
            continue
        # skip router/gate
        if should_skip(name):
            continue
        # extract prefix up to "expert" then the index
        parts = name.split(".")
        try:
            exp_pos = next(i for i, p in enumerate(parts) if "expert" in p.lower())
        except StopIteration:
            continue
        prefix = ".".join(parts[:exp_pos])
        # the expert index is the next token after "expert"
        exp_token = parts[exp_pos]  # e.g. "experts" or "expert"
        if exp_pos + 1 < len(parts):
            try:
                idx = int(parts[exp_pos + 1])
            except ValueError:
                idx = 0
        else:
            idx = 0

        if prefix not in expert_layers:
            expert_layers[prefix] = []
        expert_layers[prefix].append((idx, name, mod))

    # sort by expert index within each group
    groups = []
    for prefix, entries in expert_layers.items():
        entries.sort(key=lambda x: x[0])
        groups.append({
            "layer_path": prefix,
            "expert_indices": [e[0] for e in entries],
            "expert_modules": [(e[0], e[1], e[2]) for e in entries],
        })
    return groups


def expert_bytes_per_param(model, dtype):
    """Estimate bytes per parameter for the model's dtype."""
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 2
    elif dtype == torch.float32:
        return 4
    return 2


def expert_param_count(module):
    """Count parameters in a single expert module."""
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# GPU memory management for 2x T4
# ---------------------------------------------------------------------------

def get_gpu_info():
    """Return list of (index, name, free_gb, total_gb)."""
    devs = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        devs.append((i, name, free / (1024**3), total / (1024**3)))
    return devs


def get_total_free_vram_gb():
    """Total free VRAM across all GPUs in GB."""
    total = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        total += free
    return total / (1024**3)


def calc_batch_size(expert_groups, dtype, safety=0.6):
    """
    Calculate how many experts per batch to quantize at once.
    safety=0.6 means use only 60% of free VRAM (leave headroom).
    Returns (batch_size, expert_bytes).
    """
    if not expert_groups:
        return 1, 0

    # pick first expert as reference for size
    ref_module = expert_groups[0]["expert_modules"][0][2]
    ref_params = expert_param_count(ref_module)
    bpp = expert_bytes_per_param(None, dtype)
    expert_bytes = ref_params * bpp  # bytes for one expert in fp16/bf16

    # quantized version will be smaller, but we need fp16 for the quantize op
    # so we need space for: original + quantized copy (in-place, so ~1x)
    needed_per_expert = expert_bytes  # ~1x for in-place quantization

    free_gb = get_total_free_vram_gb()
    usable_gb = free_gb * safety
    usable_bytes = usable_gb * (1024**3)

    if needed_per_expert == 0:
        return 1, 0

    batch = max(1, int(usable_bytes // needed_per_expert))
    return batch, expert_bytes


# ---------------------------------------------------------------------------
# expert batched quantization
# ---------------------------------------------------------------------------

def quantize_experts_batched(model, quant_fn, dtype, batch_size=None, device="cuda:0"):
    """
    Quantize MoE experts in batches that fit in VRAM.
    For each batch: load experts to GPU -> quantize -> offload to CPU.
    """
    groups = find_expert_groups(model)
    if not groups:
        print("  no MoE experts found, doing full model PTQ")
        return ptq(model, quant_fn)

    total_experts = sum(len(g["expert_modules"]) for g in groups)
    total_layers = len(groups)
    print(f"  found {total_experts} experts across {total_layers} layers")

    if batch_size is None:
        batch_size, expert_bytes = calc_batch_size(groups, dtype)
        print(f"  auto batch size: {batch_size} experts at a time")
        print(f"  each expert: ~{expert_bytes / (1024**2):.1f} MB (fp16)")
    else:
        print(f"  user batch size: {batch_size} experts at a time")

    # process layer by layer
    quantized_count = 0
    skipped_count = 0

    for layer_idx, group in enumerate(groups):
        experts = group["expert_modules"]  # [(idx, name, module), ...]
        n_experts = len(experts)

        print(f"\n  layer {layer_idx+1}/{total_layers} ({group['layer_path']})  "
              f"{n_experts} experts")

        # process in batches
        for batch_start in range(0, n_experts, batch_size):
            batch_end = min(batch_start + batch_size, n_experts)
            batch = experts[batch_start:batch_end]
            batch_indices = [b[0] for b in batch]

            print(f"    batch [{batch_start}:{batch_end}]  experts {batch_indices}")

            # 1. move batch to GPU
            batch_modules = []
            for idx, name, module in batch:
                mod = module.to(device)
                batch_modules.append((idx, name, mod))

            # 2. quantize each expert in the batch
            for idx, name, mod in batch_modules:
                for pname, param in mod.named_parameters():
                    if "bias" in pname:
                        continue
                    full_name = f"{name}.{pname}"
                    with torch.no_grad():
                        param.copy_(quant_fn(param))
                quantized_count += 1

            # 3. offload back to CPU
            for idx, name, mod in batch_modules:
                mod.to("cpu")

            # 4. free GPU memory
            del batch_modules
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n  quantized {quantized_count} experts, skipped {skipped_count}")
    return model


# ---------------------------------------------------------------------------
# PTQ (no teacher, no MoE or simple mode)
# ---------------------------------------------------------------------------

def ptq(model, quant_fn):
    moe_flag = is_moe(model)
    n, sk = 0, 0
    for name, p in model.named_parameters():
        if "bias" in name:
            continue
        if moe_flag and should_skip(name):
            sk += 1
            continue
        with torch.no_grad():
            p.copy_(quant_fn(p))
            n += 1
    print(f"  quantized {n} layers, skipped {sk} router layers")
    return model


# ---------------------------------------------------------------------------
# Distillation with expert batching
# ---------------------------------------------------------------------------

class CalibDS(Dataset):
    def __init__(self, tok, texts, max_len):
        enc = tok(texts, truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="pt")
        self.enc = enc
        self.labels = enc["input_ids"].clone()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


def distill(student, teacher, tok, quant_fn, calib, epochs, bs, lr, max_len, dev):
    teacher.eval()
    student.train()
    moe_flag = is_moe(student)

    for p in teacher.parameters():
        p.requires_grad = False

    # PTQ init
    if moe_flag:
        groups = find_expert_groups(student)
        if groups:
            # just init expert weights in-place, no GPU needed
            for group in groups:
                for idx, name, mod in group["expert_modules"]:
                    for pname, param in mod.named_parameters():
                        if "bias" in pname:
                            continue
                        with torch.no_grad():
                            param.copy_(quant_fn(param))
        # also quantize non-expert layers
        for name, p in student.named_parameters():
            if "bias" in name:
                continue
            if should_skip(name):
                continue
            if "expert" in name.lower():
                continue
            with torch.no_grad():
                p.copy_(quant_fn(p))
    else:
        for name, p in student.named_parameters():
            if "bias" in name:
                continue
            with torch.no_grad():
                p.copy_(quant_fn(p))

    ds = CalibDS(tok, calib, max_len)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)

    student.to(dev)
    teacher.to(dev)

    for ep in range(epochs):
        total = 0.0
        for batch, _ in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.no_grad():
                t_logits = teacher(**batch).logits

            # re-quantize for STE
            for name, p in student.named_parameters():
                if "bias" in name:
                    continue
                if moe_flag and should_skip(name):
                    continue
                p.data.copy_(quant_fn(p.detach()))

            s_logits = student(**batch).logits
            loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        print(f"  epoch {ep+1}/{epochs}  loss={avg:.4f}")

    student.cpu()
    teacher.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return student


# ---------------------------------------------------------------------------
# Unsloth model loading — dual T4 optimized
# ---------------------------------------------------------------------------

def load_with_unsloth(model_id, dtype, load_4bit, device_map="auto"):
    """Load via Unsloth with automatic dual-T4 + CPU offload."""
    try:
        from unsloth import FastLanguageModel, FastModel
    except ImportError:
        print("  unsloth not found, falling back to transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            offload_folder="offload" if device_map == "auto" else None,
            load_in_4bit=load_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer, False

    # FastModel first (MoE support) — may force fp32 for some models
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=4096,
            load_in_4bit=load_4bit,
            load_in_8bit=False,
            dtype=dtype,
            trust_remote_code=True,
        )
        # check if it loaded in fp32 and would OOM — detect total param GB
        total_gb = sum(p.numel() for p in model.parameters()) * 4 / (1024**3)
        free_gb = get_total_free_vram_gb()
        if total_gb > free_gb:
            print(f"  FastModel forced fp32 ({total_gb:.0f} GB), OOM risk — "
                  f"retrying with FastLanguageModel 4-bit...")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raise MemoryError("fp32 too large")
        return model, tokenizer, True
    except Exception:
        # FastLanguageModel fallback — handles 4-bit better
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=4096,
                load_in_4bit=load_4bit,
                dtype=dtype,
                trust_remote_code=True,
            )
            return model, tokenizer, True
        except Exception:
            print("  unsloth loading failed, falling back to transformers 4-bit...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                load_in_4bit=True,
                offload_folder="offload",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            return model, tokenizer, False


def load_teacher_unsloth(model_id, dtype, load_4bit=False):
    """Load teacher model — tries fp16 first, falls back to 4-bit if OOM."""
    try:
        from unsloth import FastLanguageModel, FastModel
        for attempt in range(2):
            use_4bit = load_4bit or (attempt == 1)
            kw = dict(
                model_name=model_id, max_seq_length=4096,
                load_in_4bit=use_4bit, dtype=dtype,
                trust_remote_code=True,
                offload_folder="offload",
            )
            try:
                model, tok = FastModel.from_pretrained(**kw)
            except Exception:
                model, tok = FastLanguageModel.from_pretrained(**kw)
            # check for fp32 OOM
            total_gb = sum(p.numel() for p in model.parameters()) * 4 / (1024**3)
            free_gb = get_total_free_vram_gb()
            if total_gb > free_gb:
                print(f"  teacher fp32 ({total_gb:.0f} GB) too large, retrying 4-bit...")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                continue
            return model, tok
        # fall through — even 4-bit failed somehow
        raise RuntimeError("teacher loading failed after 2 attempts")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True, load_in_4bit=True,
            offload_folder="offload",
        )
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tok


# ---------------------------------------------------------------------------
# interactive prompts
# ---------------------------------------------------------------------------

def ask(prompt, default=None, options=None):
    suffix = ""
    if options:
        suffix = f" [{'/'.join(options)}]"
    if default:
        suffix += f" (default: {default})"
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if not val and default:
            return default
        if not options or val in options:
            return val
        print(f"  invalid, choose from: {options}")


def pick_device():
    devs = get_gpu_info()
    if not devs:
        print("  no GPU found, using CPU")
        return "cpu"
    print("  available GPUs:")
    for i, name, free, total in devs:
        print(f"    [{i}] {name}  ({free:.1f}/{total:.1f} GB free)")
    idx = ask("  pick primary GPU", default="0", options=[str(d[0]) for d in devs])
    return f"cuda:{idx}"


def hardware_check(load_4bit):
    """Print hardware budget."""
    print("\n--- hardware budget ---")
    gpu_info = get_gpu_info()
    for i, name, free, total in gpu_info:
        print(f"  GPU {i}: {name}  {free:.1f} GB free / {total:.1f} GB total")
    total_free = get_total_free_vram_gb()
    print(f"  total free VRAM: {total_free:.1f} GB")
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  CPU RAM: {ram.available / (1024**3):.1f} GB free / {ram.total / (1024**3):.1f} GB total")
    except ImportError:
        print("  CPU RAM: (install psutil for RAM info)")
    if load_4bit:
        print("  4-bit loading: model uses ~0.5 bytes/param on GPU, rest offloaded to CPU")
    else:
        print("  16-bit loading: model uses ~2 bytes/param on GPU, rest offloaded to CPU")
    print("  expert batching: loads N experts at a time, quantizes, offloads to CPU")
    print()


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

PRESETS = {
    "qwen35b": {
        "name": "Qwen3.6-35B-A3B (MoE)",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "teacher": "Qwen/Qwen3.6-35B-A3B",
        "mode": "ternary",
        "load_4bit": True,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "qwen30b": {
        "name": "Qwen3-30B-A3B (MoE)",
        "model": "Qwen/Qwen3-30B-A3B",
        "teacher": "Qwen/Qwen3-30B-A3B",
        "mode": "ternary",
        "load_4bit": True,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "qwen35_08b": {
        "name": "Qwen3.5-0.8B (dense, small)",
        "model": "Qwen/Qwen3.5-0.8B",
        "teacher": "Qwen/Qwen3.5-0.8B",
        "mode": "ternary",
        "load_4bit": False,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "gemma4_31b": {
        "name": "Gemma 4 31B (dense)",
        "model": "google/gemma-4-31B",
        "teacher": "google/gemma-4-31B",
        "mode": "ternary",
        "load_4bit": True,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "gemma4_26b": {
        "name": "Gemma 4 26B-A4B (MoE)",
        "model": "google/gemma-4-26B-A4B",
        "teacher": "google/gemma-4-26B-A4B",
        "mode": "ternary",
        "load_4bit": True,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "llama8b": {
        "name": "Llama-3-8B (dense)",
        "model": "meta-llama/Llama-3-8B",
        "teacher": "meta-llama/Llama-3-8B",
        "mode": "ternary",
        "load_4bit": True,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "qwen05b": {
        "name": "Qwen2.5-0.5B (dense, quick test)",
        "model": "Qwen/Qwen2.5-0.5B",
        "teacher": "Qwen/Qwen2.5-0.5B",
        "mode": "ternary",
        "load_4bit": False,
        "dtype": "fp16",
        "expert_batch": None,
    },
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _is_interactive():
    """Check if we can use input() (not in Jupyter/Kaggle/non-TTY)."""
    import sys
    return sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False


def _apply_preset(preset_key):
    """Apply a preset and return config dict."""
    p = PRESETS[preset_key]
    print(f"\n  preset: {p['name']}")
    mode_name = p["mode"]
    quant_fn = onebit_weight if mode_name == "1bit" else ternary_weight
    return {
        "model_id": p["model"],
        "teacher_id": p.get("teacher", ""),
        "mode_name": mode_name,
        "quant_fn": quant_fn,
        "load_4bit": p["load_4bit"],
        "manual_batch": p["expert_batch"],
        "out_dir": f"{p['model'].split('/')[-1]}-{mode_name}",
        "dtype": {"fp16": torch.float16, "bf16": torch.bfloat16}[p["dtype"]],
    }


def main():
    parser = argparse.ArgumentParser(description="1-bit / ternary quantizer")
    parser.add_argument("--model", "-m", default=None, help="student model (HF name or path)")
    parser.add_argument("--download", default=None,
                        help="download a model to HF cache (use preset name or HF model ID)")
    parser.add_argument("--dequantize", default=None,
                        help="dequantize a packed model directory back to FP16")
    parser.add_argument("--gguf", default=None,
                        help="convert a packed model directory to GGUF format")
    parser.add_argument("--preset", choices=list(PRESETS.keys()) + ["list"], default=None,
                        help="use a preset config (use 'list' to show available presets)")
    parser.add_argument("--teacher", "-t", default=None, help="teacher model (enables distillation)")
    parser.add_argument("--mode", choices=["1bit", "ternary"], default=None, help="quantization mode")
    parser.add_argument("--output", "-o", default=None, help="output directory")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default=None)
    parser.add_argument("--load-4bit", action="store_true", default=None, help="load student in 4-bit")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="distillation batch size")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--calib-file", default=None, help="calibration text file")
    parser.add_argument("--device", default=None, help="device (cuda:0, cpu, etc)")
    parser.add_argument("--expert-batch", type=int, default=None, help="MoE experts per batch")
    args, _ = parser.parse_known_args()

    # --preset list
    if args.preset == "list":
        print("\navailable presets:")
        for k, v in PRESETS.items():
            print(f"  {k:12s}  {v['model']}  ({v['mode']})")
        return

    # --download: pre-download model to HF cache
    if args.download:
        dl_id = PRESETS[args.download]["model"] if args.download in PRESETS else args.download
        print(f"downloading {dl_id} to HF cache...")
        print(f"  cache: {os.environ['HF_HOME']}")
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        AutoConfig.from_pretrained(dl_id, trust_remote_code=True)
        AutoTokenizer.from_pretrained(dl_id, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            dl_id, torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        print(f"  done — cached at {os.environ['HF_HOME']}/hub/")
        return

    # --dequantize
    if args.dequantize:
        output = args.output or f"{args.dequantize.rstrip('/').split('/')[-1]}-dequantized"
        dequantize_model(args.dequantize, output)
        return

    # --gguf
    if args.gguf:
        output = args.output or f"{args.gguf.rstrip('/').split('/')[-1]}.gguf"
        save_gguf(args.gguf, output)
        return

    interactive = _is_interactive() and not args.preset

    print("=" * 60)
    print("  1-bit / ternary quantizer (Unsloth + dual T4)")
    print("=" * 60)

    # --- preset or interactive mode ---
    if args.preset:
        cfg = _apply_preset(args.preset)
        model_id = cfg["model_id"]
        teacher_id = cfg["teacher_id"]
        mode_name = cfg["mode_name"]
        quant_fn = cfg["quant_fn"]
        load_4bit = cfg["load_4bit"]
        manual_batch = cfg["manual_batch"]
        out_dir = cfg["out_dir"]
        dtype = cfg["dtype"]
    else:
        # defaults for non-interactive (args.model) case
        teacher_id = ""
        mode_name = "ternary"
        quant_fn = ternary_weight
        load_4bit = False
        manual_batch = None
        dtype = torch.float16
        out_dir = None

        if args.model:
            model_id = args.model
        elif interactive:
            print("\npresets:")
            keys = list(PRESETS.keys())
            for i, k in enumerate(keys):
                print(f"  [{i+1}] {PRESETS[k]['name']}")
            print(f"  [{len(keys)+1}] custom")
            pidx = ask("pick", default="1", options=[str(i+1) for i in range(len(keys)+1)])
            pidx = int(pidx)
            if 1 <= pidx <= len(keys):
                cfg = _apply_preset(keys[pidx - 1])
                model_id = cfg["model_id"]
                teacher_id = cfg["teacher_id"]
                mode_name = cfg["mode_name"]
                quant_fn = cfg["quant_fn"]
                load_4bit = cfg["load_4bit"]
                manual_batch = cfg["manual_batch"]
                out_dir = cfg["out_dir"]
                dtype = cfg["dtype"]
            else:
                model_id = ask("\nstudent model (HF name or local path)")
                teacher_id = ask("teacher model (HF name/path, or empty for PTQ)", default="")
                print("\nquantization mode:")
                print("  [1] 1bit    — binary {-1, +1}")
                print("  [2] ternary — ternary {-1, 0, +1}")
                mode_key = ask("pick", default="2", options=["1", "2"])
                mode_name, quant_fn = QUANT[mode_key]
                load_4bit_str = ask("load student in 4-bit first? (faster, less VRAM)", default="y", options=["y", "n"])
                load_4bit = load_4bit_str == "y"
                print("\nexpert batch mode (for MoE):")
                print("  [a] auto — fit as many experts as VRAM allows")
                print("  [m] manual — choose batch size yourself")
                batch_mode = ask("pick", default="a", options=["a", "m"])
                manual_batch = None
                if batch_mode == "m":
                    manual_batch = int(ask("experts per batch", default="4"))
                out_dir = ask("output directory", default=f"{model_id.split('/')[-1]}-{mode_name}")
                dtype_str = ask("dtype", default="fp16", options=["fp16", "bf16"])
                dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
        else:
            print("ERROR: --model is required in non-interactive mode")
            parser.print_help()
            sys.exit(1)

    # --- CLI args override preset/interactive values ---
    if args.teacher is not None:
        teacher_id = args.teacher
    if args.mode:
        mode_name = "1bit" if args.mode == "1bit" else "ternary"
        quant_fn = onebit_weight if args.mode == "1bit" else ternary_weight
    if args.load_4bit is not None:
        load_4bit = args.load_4bit
    if args.expert_batch is not None:
        manual_batch = args.expert_batch
    if args.output:
        out_dir = args.output
    elif out_dir is None:
        out_dir = f"{model_id.split('/')[-1]}-{mode_name}"
    if args.dtype:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    calib_texts = None
    epochs = args.epochs or 3
    lr = args.lr or 5e-4
    batch_size = args.batch_size or 1
    max_len = args.max_len or 512

    if teacher_id:
        if interactive and not args.device:
            print("\n--- distillation settings ---")
            device = pick_device()
            epochs = int(ask("epochs", default=str(epochs)))
            lr = float(ask("learning rate", default=str(lr)))
            batch_size = int(ask("batch size", default=str(batch_size)))
            max_len = int(ask("max sequence length", default=str(max_len)))

        calib_path = args.calib_file
        if interactive and not calib_path:
            calib_path = ask("calib file (one sentence/line, or empty for dummy)", default="")

        if calib_path and os.path.isfile(calib_path):
            calib_texts = Path(calib_path).read_text().strip().splitlines()
            print(f"  loaded {len(calib_texts)} calibration lines")
        elif calib_path:
            print(f"  file not found: {calib_path}, using dummy calibration")
        if not calib_texts:
            calib_texts = ["Hello, this is a calibration sentence."] * 64

    # --- load ---
    hardware_check(load_4bit)

    print(f"loading student: {model_id}")
    student, tokenizer, used_unsloth = load_with_unsloth(model_id, dtype, load_4bit)
    moe_flag = is_moe(student)
    print(f"  MoE detected: {moe_flag}")

    if moe_flag:
        groups = find_expert_groups(student)
        total_experts = sum(len(g["expert_modules"]) for g in groups)
        print(f"  found {total_experts} experts across {len(groups)} layers")

        auto_bs, expert_bytes = calc_batch_size(groups, dtype)
        print(f"  auto batch size: {auto_bs} experts/iter (~{expert_bytes/(1024**2):.1f} MB each)")

    if teacher_id:
        # estimate if both models can fit: student (4-bit ≈ 0.5B) + teacher (≈ student)
        # rough check: total_params * bytes_per_param > free_vram + free_ram
        total_params = sum(p.numel() for p in student.parameters())
        student_gb = total_params * 0.5 / (1024**3)  # 4-bit ≈ 0.5 bytes/param
        teacher_gb = total_params * 0.5 / (1024**3)  # same estimate
        free_vram = get_total_free_vram_gb()
        try:
            import psutil
            free_ram = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            free_ram = 20  # guess

        if student_gb + teacher_gb > free_vram + free_ram * 0.7:
            print(f"  not enough memory for teacher+student "
                  f"(need ~{student_gb+teacher_gb:.0f} GB, have "
                  f"{free_vram:.0f} VRAM + {free_ram:.0f} RAM). "
                  f"falling back to PTQ-only.")
            teacher = None
        else:
            print(f"loading teacher: {teacher_id}")
            try:
                teacher, _ = load_teacher_unsloth(teacher_id, dtype)
            except Exception as e:
                print(f"  teacher loading failed ({e}). falling back to PTQ-only.")
                teacher = None
            # clean up any partial offload
            import shutil
            shutil.rmtree("offload", ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if teacher is not None:
            print("starting distillation...")
            student = distill(student, teacher, tokenizer, quant_fn,
                              calib_texts, epochs, batch_size, lr, max_len, device)
        else:
            print("running PTQ (no teacher)...")
            if moe_flag:
                student = quantize_experts_batched(
                    student, quant_fn, dtype,
                    batch_size=manual_batch,
                    device=device,
                )
            else:
                student = ptq(student, quant_fn)
    elif moe_flag:
        print("running expert-batched PTQ...")
        student = quantize_experts_batched(
            student, quant_fn, dtype,
            batch_size=manual_batch,
            device=device,
        )
    else:
        print("running PTQ...")
        student = ptq(student, quant_fn)

    # --- pack and save ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Pack directly from the already-quantized parameters.
    # (Weights have been quantized in-place by ptq / quantize_experts_batched / distill.)
    print("packing quantized weights...")
    packed_state = {}
    pack_config = {}
    gs = 128  # group size for both 1-bit and ternary (matches Bonsai Q1_0_g128 / Q2_0)

    for name, param in student.named_parameters():
        if "bias" in name:
            continue
        if moe_flag and should_skip(name):
            # router / gate layers kept in fp16
            packed_state[name] = param.detach().cpu().half()
            pack_config[name] = "fp16"
            continue

        w = param.detach().cpu().float()
        n_w = w.numel()
        # pad to multiple of group_size for scale computation
        pad = (gs - n_w % gs) % gs
        w_flat = w.view(-1)
        if pad:
            w_flat = F.pad(w_flat, (0, pad))
        w_blocks = w_flat.view(-1, gs)

        if mode_name == "1bit":
            scales = w_blocks.abs().max(dim=1).values.clamp(min=1e-8)  # Bonsai absmax
            packed_state[name] = pack_q1_0_blocks(w, scales, gs)
            pack_config[name] = "1bit"
        else:  # ternary
            scales = w_blocks.abs().max(dim=1).values.clamp(min=1e-8)
            # re-derive codes from quantized values: w / scale rounded to {-1,0,+1}
            q = (w_blocks / scales.unsqueeze(1)).round().to(torch.int32).clamp(-1, 1)
            codes = q + 1  # shift to {0, 1, 2} for 2-bit storage
            packed_state[name] = pack_q2_0_blocks(codes, scales, gs)
            pack_config[name] = "ternary"

    # save packed weights
    torch.save(packed_state, Path(out_dir) / "quantized_weights.pt")
    torch.save({
        "pack_config": pack_config,
        "mode": mode_name,
        "group_size": gs,
    }, Path(out_dir) / "quant_config.pt")

    # save model config + tokenizer (for reloading)
    student.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\ndone — saved to {out_dir}")
    print(f"  config + tokenizer → {out_dir}/")
    print(f"  packed weights      → {out_dir}/quantized_weights.pt")

    # show compression
    orig_size = sum(p.numel() for p in student.parameters()) * 2  # fp16
    packed_path = Path(out_dir) / "quantized_weights.pt"
    packed_size = packed_path.stat().st_size
    ratio = orig_size / packed_size if packed_size > 0 else float("inf")
    print(f"  compression: {orig_size/1024**3:.1f} GB (fp16) → "
          f"{packed_size/1024**3:.2f} GB ({ratio:.0f}x)")


if __name__ == "__main__":
    main()
