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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# quantization primitives
# ---------------------------------------------------------------------------

def ternary_weight(w):
    w_fp32 = w.float()
    scale = w_fp32.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    w_norm = w_fp32 / scale
    w_t = w_norm.tanh()
    w_t = torch.where(w_t > 0.4, torch.ones_like(w_t), w_t)
    w_t = torch.where(w_t < -0.4, -torch.ones_like(w_t), w_t)
    w_t = torch.where(w_t.abs() < 0.4, torch.zeros_like(w_t), w_t)
    return w_t.to(w.dtype)


def onebit_weight(w):
    return w.float().sign().to(w.dtype)


QUANT = {"1": ("1bit", onebit_weight), "2": ("ternary", ternary_weight)}


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
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer, False

    # Try FastModel first (supports MoE), fall back to FastLanguageModel
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=4096,
            load_in_4bit=load_4bit,
            load_in_8bit=False,
            dtype=dtype,
            trust_remote_code=True,
        )
        return model, tokenizer, True
    except Exception:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=4096,
            load_in_4bit=load_4bit,
            dtype=dtype,
            trust_remote_code=True,
        )
        return model, tokenizer, True


def load_teacher_unsloth(model_id, dtype):
    """Load teacher model — fp16, no 4-bit, auto device_map for 2x T4."""
    try:
        from unsloth import FastLanguageModel, FastModel
        try:
            model, tok = FastModel.from_pretrained(
                model_name=model_id, max_seq_length=4096,
                load_in_4bit=False, dtype=dtype,
                trust_remote_code=True,
            )
            return model, tok
        except Exception:
            model, tok = FastLanguageModel.from_pretrained(
                model_name=model_id, max_seq_length=4096,
                load_in_4bit=False, dtype=dtype,
                trust_remote_code=True,
            )
            return model, tok
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True,
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
        "mode": "ternary",
        "load_4bit": False,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "qwen30b": {
        "name": "Qwen3-30B-A3B (MoE)",
        "model": "Qwen/Qwen3-30B-A3B",
        "mode": "ternary",
        "load_4bit": False,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "llama8b": {
        "name": "Llama-3-8B (dense)",
        "model": "meta-llama/Llama-3-8B",
        "mode": "ternary",
        "load_4bit": True,
        "dtype": "fp16",
        "expert_batch": None,
    },
    "qwen05b": {
        "name": "Qwen2.5-0.5B (dense, quick test)",
        "model": "Qwen/Qwen2.5-0.5B",
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
        "teacher_id": "",
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
        print(f"loading teacher: {teacher_id}")
        teacher, _ = load_teacher_unsloth(teacher_id, dtype)
        print("starting distillation...")
        student = distill(student, teacher, tokenizer, quant_fn,
                          calib_texts, epochs, batch_size, lr, max_len, device)
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

    # --- save ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"\ndone — saved to {out_dir}")


if __name__ == "__main__":
    main()
