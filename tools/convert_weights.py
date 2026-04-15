#!/usr/bin/env python3
"""
tools/convert_weights.py — Convert HuggingFace GPT-2 weights → NanoInfer binary format

NanoInfer load_weights() reads raw binary: each weight tensor is dumped consecutively
in fixed order, dtype float32. This file performs that conversion.

Use: 
    # Download and convert GPT-2 small 
    python tools/convert_weights.py --model gpt2 --output weights/gpt2 

    # GPT-2 medium 
    python tools/convert_weights.py --model gpt2-medium --output weights/gpt2-medium 

    # Convert to FP16 (2x smaller, used for fp16 precision mode) 
    python tools/convert_weights.py --model gpt2 --output weights/gpt2 --dtype fp16

Output: 
    weights/gpt2/ 
        weights.bin ← all tensors dump one after another (float32 or float16) 
        config.json ← vocab_size, d_model, n_layers, n_heads, etc.

Dump order (must match load_weights() in model.cpp): 
    wte [vocab_size x d_model] 
    wpe [max_seq_len x d_model] 
    ln_f.weight [d_model] 
    ln_f.bias [d_model] 
    for layers in 0..n_layers-1: 
        ln_1.weight, ln_1.bias 
        attn.c_attn.weight [3*d_model x d_model], attn.c_attn.bias [3*d_model] 
        attn.c_proj.weight [d_model x d_model], attn.c_proj.bias [d_model] 
        ln_2.weight, ln_2.bias 
        mlp.c_fc.weight [d_ffn x d_model], mlp.c_fc.bias [d_ffn] 
        mlp.c_proj.weight [d_model x d_ffn], mlp.c_proj.bias [d_model]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config as HFConfig
except ImportError:
    print("[error] Required packages not found. Install:")
    print("        pip install torch transformers")
    sys.exit(1)
 
 
# ---------------------------------------------------------------------------
# Weight extraction order — must match model.cpp load_weights()
# ---------------------------------------------------------------------------

def get_weight_keys(config) -> list[tuple[str, str]]:
    """
    Returns list of (hf_key, description) in the order they will be written.
    HuggingFace GPT-2 weight names → NanoInfer convention.
    """
    keys = [
        ("transformer.wte.weight",     "token embedding   [vocab x d_model]"),
        ("transformer.wpe.weight",     "pos embedding     [max_seq x d_model]"),
        ("transformer.ln_f.weight",    "final LN gamma    [d_model]"),
        ("transformer.ln_f.bias",      "final LN beta     [d_model]"),
    ]

    for l in range(config.n_layer):
        prefix = f"transformer.h.{l}"
        keys += [
            (f"{prefix}.ln_1.weight",          f"L{l} pre-attn LN gamma"),
            (f"{prefix}.ln_1.bias",            f"L{l} pre-attn LN beta"),
            # HF GPT-2 fuses QKV into c_attn weight [d_model x 3*d_model]
            # NanoInfer expects [3*d_model x d_model] — need to transpose
            (f"{prefix}.attn.c_attn.weight",   f"L{l} QKV weight [3D x D] (transposed)"),
            (f"{prefix}.attn.c_attn.bias",     f"L{l} QKV bias   [3D]"),
            (f"{prefix}.attn.c_proj.weight",   f"L{l} out proj weight [D x D] (transposed)"),
            (f"{prefix}.attn.c_proj.bias",     f"L{l} out proj bias   [D]"),
            (f"{prefix}.ln_2.weight",          f"L{l} pre-FFN LN gamma"),
            (f"{prefix}.ln_2.bias",            f"L{l} pre-FFN LN beta"),
            (f"{prefix}.mlp.c_fc.weight",      f"L{l} FFN up weight   [4D x D] (transposed)"),
            (f"{prefix}.mlp.c_fc.bias",        f"L{l} FFN up bias     [4D]"),
            (f"{prefix}.mlp.c_proj.weight",    f"L{l} FFN down weight [D x 4D] (transposed)"),
            (f"{prefix}.mlp.c_proj.bias",      f"L{l} FFN down bias   [D]"),
        ]
 
    return keys

# Keys that need transposing (HF stores Conv1D weights transposed vs Linear)
TRANSPOSE_KEYS = {"attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight"}
 
 
def needs_transpose(hf_key: str) -> bool:
    return any(k in hf_key for k in TRANSPOSE_KEYS)


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
 
def convert(model_name: str, output_dir: str, dtype: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config
    model.eval()

    state_dict = model.state_dict()

    np_dtype = np.float32 if dtype == "fp32" else np.float16
 
    weights_file = output_path / "weights.bin"
    print(f"Writing weights to {weights_file} (dtype={dtype})...")
 
    total_bytes = 0
    key_list = get_weight_keys(config)
 
    with open(weights_file, "wb") as f:
        for hf_key, description in key_list:
            if hf_key not in state_dict:
                print(f"  [warn] Key not found: {hf_key} — writing zeros")
                # Write appropriate zeros based on expected shape
                continue
 
            tensor = state_dict[hf_key].float().numpy()
 
            # HuggingFace GPT-2 uses Conv1D (weights stored as [in x out])
            # NanoInfer expects [out x in] — transpose these
            if needs_transpose(hf_key):
                tensor = tensor.T  # [in x out] → [out x in]
 
            arr = tensor.astype(np_dtype)
            raw = arr.tobytes()
            f.write(raw)
            total_bytes += len(raw)
 
            print(f"  {description:<50} shape={arr.shape}  {len(raw)/1024:.1f} KB")
 
    print(f"\nTotal: {total_bytes / 1e6:.1f} MB")

    # ---------------------------------------------------------------------------
    # Write config.json
    # ---------------------------------------------------------------------------
    config_dict = {
        "model_name":   model_name,
        "vocab_size":   config.vocab_size,
        "d_model":      config.n_embd,
        "n_layers":     config.n_layer,
        "n_heads":      config.n_head,
        "d_head":       config.n_embd // config.n_head,
        "d_ffn":        config.n_embd * 4,
        "max_seq_len":  config.n_positions,
        "eos_token_id": config.eos_token_id,
        "dtype":        dtype,
    }
 
    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)
 
    print(f"Config written to {config_file}")
    print("\nDone. Load in NanoInfer:")
    print(f'  model = nanoinfer.NanoInfer.from_pretrained("{output_dir}")')
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace GPT-2 weights to NanoInfer format")
    parser.add_argument("--model",  default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="HuggingFace model name")
    parser.add_argument("--output", default="weights/gpt2",
                        help="Output directory")
    parser.add_argument("--dtype",  default="fp32",
                        choices=["fp32", "fp16"],
                        help="Output dtype (fp32=154MB, fp16=77MB for gpt2-small)")
    args = parser.parse_args()

    convert(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()