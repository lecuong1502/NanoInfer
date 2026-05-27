# Python API Reference

Complete reference for the `nanoinfer` Python package (v0.1.0).

The package exposes two public surfaces:

| Surface | Access | Purpose |
|---|---|---|
| `NanoInfer` | `from nanoinfer import NanoInfer` | High-level GPT-2 inference (HuggingFace-compatible) |
| `kernels` | `import nanoinfer.kernels as kernels` | Low-level CUDA kernels for research and benchmarking |

---

## `NanoInfer`

High-level inference class. Wraps the full GPT-2 pipeline — tokenization, forward pass, KV-cache, and sampling — behind a simple API.

```python
from nanoinfer import NanoInfer
```

---

### `NanoInfer.__init__(model_path, precision, max_seq_len, kvcache_pages)`

Load model weights from a local directory.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | *(required)* | Path to directory containing model weight files |
| `precision` | `str` | `"fp32"` | Compute precision: `"fp32"`, `"fp16"`, or `"int8"` |
| `max_seq_len` | `int` | `2048` | Maximum sequence length; determines KV-cache size |
| `kvcache_pages` | `int` | `256` | Number of pages to pre-allocate in the paged KV-cache |

**Example**

```python
model = NanoInfer("/path/to/gpt2-weights", precision="fp16", max_seq_len=1024)
```

---

### `NanoInfer.from_pretrained(model_name, precision, max_seq_len, kvcache_pages)` *(classmethod)*

Load a model by name or local path. This is the recommended constructor.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | Model name (e.g. `"gpt2"`) or local directory path |
| `precision` | `str` | `"fp32"` | `"fp32"`, `"fp16"`, or `"int8"` |
| `max_seq_len` | `int` | `2048` | Maximum sequence length |
| `kvcache_pages` | `int` | `256` | KV-cache page budget |

**Returns** `NanoInfer` — a ready-to-use inference instance.

**Example**

```python
# Load GPT-2 small with INT8 quantization
model = NanoInfer.from_pretrained("gpt2", precision="int8")

# Equivalent with explicit path
model = NanoInfer.from_pretrained("./weights/gpt2", precision="fp16")
```

---

### `NanoInfer.generate(prompt, max_tokens, temperature, top_p, top_k, seed)`

Generate a text continuation from a prompt. Uses autoregressive token generation with a KV-cache.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | *(required)* | Input text |
| `max_tokens` | `int` | `100` | Maximum number of *new* tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature. `0` = greedy (deterministic), `1.0` = standard, `>1.0` = more random |
| `top_p` | `float` | `0.95` | Nucleus sampling threshold (0–1). `1.0` disables nucleus filtering |
| `top_k` | `int` | `50` | Vocabulary cutoff; keep only top-k tokens before sampling. `0` disables |
| `seed` | `int` | `-1` | RNG seed for reproducible output. `-1` = random seed |

**Returns** `str` — the generated continuation text (the original prompt is **not** included).

**Sampling pipeline (per token)**

1. Temperature scaling — `logits /= temperature`
2. Top-k filter — zero out all but the k highest-probability tokens
3. Softmax — convert logits to probabilities
4. Top-p (nucleus) filter — keep the smallest set with cumulative probability ≥ `top_p`, renormalize
5. Sample — draw one token from the filtered distribution using a fast LCG RNG

**Examples**

```python
# Standard generation
output = model.generate("Once upon a time", max_tokens=100)

# Greedy decoding (deterministic)
output = model.generate("The capital of France is", temperature=0, max_tokens=10)

# Creative, high-variance output
output = model.generate("Roses are red,", temperature=1.4, top_p=0.9, max_tokens=50)

# Reproducible output
output = model.generate("Hello world", seed=42, max_tokens=30)
```

---

### `NanoInfer.encode(text)`

Tokenize text and return a list of integer token IDs.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `text` | `str` | Input text to tokenize |

**Returns** `list[int]` — token IDs.

```python
ids = model.encode("Hello, world!")
# e.g. [15496, 11, 995, 0]
```

---

### `NanoInfer.decode(ids)`

Convert a list of token IDs back to a string.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `ids` | `list[int]` | Token IDs |

**Returns** `str` — decoded text.

```python
text = model.decode([15496, 11, 995, 0])
# "Hello, world!"
```

---

### Properties (read-only)

| Property | Type | Description |
|---|---|---|
| `model.precision` | `str` | Active precision: `"fp32"`, `"fp16"`, or `"int8"` |
| `model.vocab_size` | `int` | Vocabulary size of the loaded model (GPT-2: 50257) |
| `model.num_layers` | `int` | Number of transformer layers (GPT-2 small: 12) |
| `model.num_heads` | `int` | Number of attention heads per layer (GPT-2 small: 12) |
| `model.d_model` | `int` | Hidden dimension size (GPT-2 small: 768) |

```python
model = NanoInfer.from_pretrained("gpt2", precision="int8")
print(model)
# NanoInfer(gpt2, precision=int8, layers=12, heads=12, d_model=768)

print(model.precision)    # "int8"
print(model.vocab_size)   # 50257
print(model.num_layers)   # 12
print(model.num_heads)    # 12
print(model.d_model)      # 768
```

---

## `nanoinfer.kernels`

Low-level CUDA kernel interface. Useful for benchmarking, ablation studies, or integrating individual kernels into other pipelines.

All kernels operate on NumPy `float32` arrays. Internally, data is transferred to GPU memory, the kernel runs, and results are copied back.

```python
import nanoinfer.kernels as kernels
```

---

### `kernels.gemm(A, B)`

Tiled GEMM (General Matrix Multiplication): computes `C = A @ B` using a custom shared-memory CUDA kernel.

For matrices ≥ 512×512, a 32×32 tile kernel is selected automatically; smaller matrices use a 16×16 tile kernel. Both variants use `+1` row padding in shared memory to eliminate bank conflicts.

**Parameters**

| Name | Type | Shape | Description |
|---|---|---|---|
| `A` | `ndarray[float32]` | `(M, K)` | Left operand |
| `B` | `ndarray[float32]` | `(K, N)` | Right operand |

**Returns** `ndarray[float32]` — shape `(M, N)`.

**Raises** `RuntimeError` if inputs are not 2-D or inner dimensions mismatch.

```python
import numpy as np
import nanoinfer.kernels as kernels

A = np.random.randn(4096, 4096).astype(np.float32)
B = np.random.randn(4096, 4096).astype(np.float32)
C = kernels.gemm(A, B)   # ~13.8 TFLOPS on A100
```

---

### `kernels.softmax(x)`

Row-wise online softmax — numerically stable, single-pass, using warp-level `__shfl_down_sync` reductions.

Uses `float4` vectorized loads when `cols % 4 == 0` and a dual-stream ILP approach for maximum throughput. Block size is adaptive: 512 threads for wide rows (≥ 1024 columns), 256 threads for narrow rows.

**Parameters**

| Name | Type | Shape | Description |
|---|---|---|---|
| `x` | `ndarray[float32]` | `(rows, cols)` | Input logits |

**Returns** `ndarray[float32]` — same shape as `x`; each row sums to `1.0`.

```python
x = np.random.randn(4096, 4096).astype(np.float32)
y = kernels.softmax(x)
print(y[0].sum())   # 1.0000001 (float32 precision)
```

---

### `kernels.flash_attention(Q, K, V, causal=True)`

Flash Attention v1 — IO-aware exact attention. Uses tiled computation over SRAM blocks to reduce HBM memory usage from O(N²) to O(N) and HBM traffic from O(N²) to O(N·d).

**Implementation:** One warp (32 threads) per query row. Dot products are computed cooperatively across lane-parallel dimensions using `__shfl_down_sync`. Supports causal masking for autoregressive generation.

**Parameters**

| Name | Type | Shape | Description |
|---|---|---|---|
| `Q` | `ndarray[float32]` | `(batch, heads, seq_len, d_head)` | Query tensor |
| `K` | `ndarray[float32]` | `(batch, heads, seq_len, d_head)` | Key tensor |
| `V` | `ndarray[float32]` | `(batch, heads, seq_len, d_head)` | Value tensor |
| `causal` | `bool` | — | Apply causal mask (default `True`). Set `False` for bidirectional (BERT-style) attention |

**Returns** `ndarray[float32]` — same shape as `Q`.

```python
batch, heads, seq, d_head = 1, 12, 2048, 64
Q = np.random.randn(batch, heads, seq, d_head).astype(np.float32)
K = np.random.randn(batch, heads, seq, d_head).astype(np.float32)
V = np.random.randn(batch, heads, seq, d_head).astype(np.float32)

O = kernels.flash_attention(Q, K, V, causal=True)   # shape: (1, 12, 2048, 64)
```

---

## Package Metadata

```python
import nanoinfer

print(nanoinfer.__version__)    # "0.1.0"
print(nanoinfer.__author__)     # "lecuong1502"
print(nanoinfer.__license__)    # "MIT"
print(nanoinfer.__all__)        # ["NanoInfer", "kernels", "__version__"]
```

---

## Error Handling

| Exception | When raised |
|---|---|
| `ImportError` | C extension `.so` not found — rebuild with `pip install -e .` |
| `RuntimeError` | CUDA error (out-of-memory, unsupported GPU, dimension mismatch) |
| `ValueError` | Invalid `precision` string (must be `"fp32"`, `"fp16"`, or `"int8"`) |

```python
try:
    model = NanoInfer.from_pretrained("gpt2", precision="bf16")  # invalid
except ValueError as e:
    print(e)  # precision must be one of: fp32, fp16, int8
```
