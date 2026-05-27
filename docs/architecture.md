# Architecture

A technical deep-dive into every kernel, layer, and system component in NanoInfer.

---

## Overview

NanoInfer is a from-scratch CUDA inference engine for GPT-2 style transformer models. The codebase is organized into four layers:

```
┌─────────────────────────────────────┐
│          Python API (pybind11)       │  nanoinfer/__init__.py
│         src/bindings/python.cpp      │
├─────────────────────────────────────┤
│            Engine Layer              │  src/engine/
│   model.cpp · kvcache.cpp            │
│   sampler.cpp                        │
├─────────────────────────────────────┤
│             Layer Layer              │  src/layers/
│  linear.cu · layernorm.cu           │
│  embedding.cu · activation.cu        │
├─────────────────────────────────────┤
│            Kernel Layer              │  src/kernels/
│  gemm.cu · softmax.cu               │
│  attention.cu · quantize.cu          │
└─────────────────────────────────────┘
```

**Design principle:** Every layer of abstraction is thin. Kernels are written by hand, not auto-generated. Every optimization is explicit and measurable.

---

## Kernel Layer (`src/kernels/`)

### Tiled GEMM (`gemm.cu`)

**The problem:** Naïve GEMM reads A and B from HBM (High Bandwidth Memory, ~2 TB/s) for every multiply-add. For a K×K inner dimension, each output element requires K HBM reads — total HBM traffic is O(M·N·K).

**The solution — tiled shared memory GEMM:**

Each thread block computes one `TILE_SIZE × TILE_SIZE` output tile of C. Rather than reading A and B from HBM for each multiply-add, threads cooperatively load one tile of A and one tile of B into shared memory (~100× faster than HBM, ~4 cycle latency vs ~600 cycles). The partial dot product is then computed from shared memory, and the next tile is loaded.

```
HBM traffic reduced:  O(M·N·K)  →  O(M·N·K / TILE_SIZE)
```

**Bank conflict elimination:**

Shared memory is organized into 32 banks. When 16 threads in the same warp read different addresses in the same bank, the accesses are serialized — 16× slowdown. In matrix tiles, column reads of `As` cause exactly this pattern.

The fix is a single extra element of padding per row:

```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 shifts each row by one bank
__shared__ float Bs[TILE_SIZE][TILE_SIZE];
```

This offsets each row by one bank, preventing any two threads from hitting the same bank on a column read.

**Adaptive tile size:**

| Matrix size | Tile | Rationale |
|---|---|---|
| < 512×512 | 16×16 | Avoids register pressure overflow on small SM occupancy |
| ≥ 512×512 | 32×32 | More data reuse per tile, fewer HBM loads |

The launcher `launch_gemm_tiled()` selects the kernel automatically based on M, N, K.

**Performance:** ~80% of cuBLAS peak TFLOPS on A100 for large square matrices.

---

### Online Softmax (`softmax.cu`)

**Three variants are implemented:**

| Kernel | Passes | Numerically stable | Use case |
|---|---|---|---|
| `softmax_naive` | 2 | ❌ | Benchmark baseline only — overflows for large logits |
| `softmax_safe` | 3 | ✅ | Reference correctness check |
| `softmax_online` | 1 | ✅ | Production — used in attention and sampling |

**Online softmax algorithm:**

Standard safe softmax requires three data passes: find max → compute exp(x − max) → normalize. Online softmax merges these into a single pass by maintaining a running max `m` and a rescaled denominator `d`:

```
For each element x_i:
    m_new = max(m_old, x_i)
    d_new = d_old × exp(m_old − m_new) + exp(x_i − m_new)
```

The correction factor `exp(m_old − m_new)` rescales the old accumulator whenever the running max increases. The final result is numerically identical to the 3-pass version.

**Implementation optimizations in `softmax_online_kernel<BLOCK>`:**

1. **`float4` vectorized loads** — when `cols % 4 == 0`, each thread loads 4 floats in a single memory instruction (4× coalescing efficiency)
2. **Dual-stream ILP** — two independent `(m, d)` streams process interleaved elements, allowing the GPU to pipeline `expf()` calls without serial data dependencies
3. **Warp reduction via `__shfl_down_sync`** — merge per-thread `(m, d)` into per-warp result in ~4 cycles with no shared memory round-trip
4. **Block reduction through shared memory** — first warp merges all warp results using a second `__shfl_down_sync` pass
5. **Hardware reciprocal `__frcp_rn`** — replaces N divisions with 1 reciprocal + N multiplications

**Adaptive block size:** 512 threads for rows ≥ 1024 (better SM occupancy), 256 threads for narrower rows (more blocks in flight).

**Speedup over 3-pass:** ~3× on A100 for 4096×4096.

---

### Flash Attention v1 (`attention.cu`)

**The problem:** Naïve attention materializes the full N×N score matrix in HBM:

```
scores = Q @ K^T      # [N, N] — O(N²) memory
probs  = softmax(scores)
output = probs @ V
```

For N=2048, d=64, batch=8: **8.59 GB** of HBM just for the score matrix.

**The solution — IO-aware tiled attention:**

Flash Attention tiles the computation. Small blocks of Q, K, V are loaded into SRAM, partial attention scores are computed, and results are accumulated with an online softmax correction factor. The full N×N matrix is never materialized.

```
Memory:  O(N²) → O(N)
HBM I/O: O(N²) → O(N·d)
```

**NanoInfer's implementation — warp-per-row design:**

| Component | Value |
|---|---|
| Block layout | `BLOCK_Q=16` query rows per block |
| Thread count | `BLOCK_Q × 32 = 512` threads (1 warp per query row) |
| KV tile size | `BLOCK_KV=32` key/value rows per tile |
| SMEM usage | `(16 + 2×32) × 64 × 4 = 20 KB` per block |
| SM occupancy | 3 concurrent blocks → 100% thread occupancy |

**Per-warp dot product:** Instead of one thread computing a full `d`-dimensional dot product (64 serial multiply-adds → pipeline stalls), each warp of 32 lanes computes it cooperatively. Each lane handles `d/32 = 2` dimensions. Warp reduction via `__shfl_down_sync` gives ~32× more parallelism.

**Causal masking:** When `causal=True`, positions `(q_start + qi) < (kv_start + kj)` are masked to `-FLT_MAX` before the softmax. Early termination when the entire KV block is beyond the current query position.

**Speedup over naïve:** 3× latency + 22× memory reduction at seq_len=2048, d_head=64.

---

### INT8 Quantization (`quantize.cu`)

**Quantization formula (per-tensor symmetric):**

```
x_q = clamp(round(x / scale), -128, 127)    # float32 → int8
x   = x_q × scale                           # int8 → float32

scale = max(|x|) / 127
```

Quantization error bound: `|x − dequant(quant(x))| ≤ scale/2`

**Three kernels:**

**`quantize_kernel`** (float32 → int8)
- `float4` vectorized loads: 1 memory instruction reads 4 floats
- Uses `__float2int_rn` (hardware round-to-nearest) instead of `roundf()`
- Handles non-aligned tails with a scalar fallback

**`dequantize_kernel`** (int32 accumulator → float32)
- After INT8 GEMM, the accumulator is int32 (int8 × int8 products can overflow int8)
- `C_float = C_int32 × scale_A × scale_B`

**`int8_gemm_kernel`** (INT8 GEMM with `dp4a`)

The `dp4a` instruction computes a 4-element dot product in a single clock cycle:
```
__dp4a(a, b, c)  =  c + a[0]×b[0] + a[1]×b[1] + a[2]×b[2] + a[3]×b[3]
```
where `a`, `b` are `int8×4` values packed into a 32-bit integer.

This gives **4× theoretical throughput** over FP32 on integer tensor cores.

Layout: B is stored transposed (`[N × K]` instead of `[K × N]`) to enable coalesced column reads during the dot product. Transposition is done in preprocessing, not in the kernel.

**Throughput:** 2–3× over FP32 GEMM; matches TensorRT FP16 at the end-to-end level.

---

## Layer Layer (`src/layers/`)

### Fused LayerNorm + Linear (`linear.cu`, `layernorm.cu`)

**The problem — HBM round-trips:**

Without fusion, LayerNorm → Linear requires three HBM passes:

```
[HBM: input] → LayerNorm kernel → [HBM: normed] → Linear kernel → [HBM: output]
```

For GPT-2 hidden layers (H=768→3072, batch=8, seq=512):
```
3 × (8 × 512 × 768 × 4 bytes) = 75 MB of HBM traffic per sub-layer
```

That traffic exists only to pass data between kernels — it carries zero compute value.

**The fused kernel:**

```
[HBM: input] → LayerNorm+Linear kernel → [HBM: output]
```

```cuda
__global__ void layernorm_linear_fused(
    const float* __restrict__ input,   // [B, S, H]
    const float* __restrict__ gamma,   // [H]
    const float* __restrict__ beta,    // [H]
    const float* __restrict__ weight,  // [H_out, H]
    float* __restrict__ output,        // [B, S, H_out]
    int H, int H_out, float eps)
{
    extern __shared__ float smem[];

    // Step 1: Compute LayerNorm
    // — mean and variance via block-level warp reductions
    // — normalized activations stored in smem (never written to HBM)

    // Step 2: Linear projection
    // — reads from smem, writes to HBM exactly once
}
```

The normalized activations live in registers and shared memory throughout — they never touch HBM. HBM traffic drops from 3 passes to 1 read + 1 write: **2× reduction** in memory operations.

**Measured speedup:** 1.4× on GPT-2 hidden layers at batch=8, seq=512 on A100. The gain scales with sequence length.

**Why this matters:** Kernel fusion is the primary technique behind TensorRT's performance advantage. Every fused kernel in TensorRT's graph optimizer follows this same principle.

---

### Other Layers

| File | Kernel | Description |
|---|---|---|
| `embedding.cu` | `token_embedding` | Lookup-table gather: `hidden[i] = wte[token_id[i]]` |
| `embedding.cu` | `positional_embedding` | Add `wpe[pos + offset]` to each token's hidden state |
| `layernorm.cu` | `layernorm` | Standalone LayerNorm (used for final LN before LM head) |
| `activation.cu` | `gelu` | GELU activation (tanh approximation, matching GPT-2 training) |
| `activation.cu` | `add` | In-place elementwise addition for residual connections |
| `linear.cu` | `linear` | Unfused linear layer (with optional bias) |

---

## Engine Layer (`src/engine/`)

### GPT-2 Model (`model.cpp`)

Implements the full GPT-2 forward pass by composing layer primitives.

**Architecture (GPT-2 small, 124M parameters):**

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 50257 |
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `d_head` | 64 (= d_model / n_heads) |
| `d_ffn` | 3072 (= 4 × d_model) |
| `max_seq_len` | 1024 |

**Per-layer forward pass (Pre-LN architecture):**

```
x = x + Attention(LayerNorm₁(x))    ← fused LN+QKV projection + Flash Attention
x = x + FFN(LayerNorm₂(x))          ← fused LN+FC1 + GELU + FC2
```

**Weight naming** follows HuggingFace GPT-2 conventions (`wte`, `wpe`, `ln1`, `qkv_w`, etc.) for compatibility with existing weight conversion scripts.

**Activation buffer strategy:** All intermediate tensors (`d_hidden`, `d_normed`, `d_qkv`, `d_attn_out`, `d_ffn_mid`) are allocated once at model load time and reused across layers and inference calls. This avoids per-step GPU memory allocation overhead.

---

### Paged KV-Cache (`kvcache.cpp`)

**The problem:** Standard KV-cache pre-allocates a fixed `[layers × heads × max_seq × d_head]` buffer. For long sequences this wastes memory; for short sequences it over-allocates.

**Paged allocation:** Memory is divided into fixed-size **pages** (default: 16 tokens per page). The cache grows one page at a time as the sequence extends. When a request completes, its pages are returned to the free pool and immediately reusable by the next request — no fragmentation.

```
PagedKVCache(
    num_layers   = 12,
    num_heads    = 12,
    head_dim     = 64,
    max_pages    = 256,   // budget cap
    page_size    = 16,    // tokens per page
    bits         = 4,     // KV quantization bits
    mode         = KVQuantMode::PROD
)
```

**KV quantization:** Keys and values are optionally quantized to 4-bit before storage (`KVQuantMode::PROD`) to further reduce memory footprint. `KVQuantMode::NONE` disables quantization.

**API:**

```cpp
cache.append(layer, head, d_key, d_value);           // add one token's KV
cache.retrieve(layer, head, seq_len, d_key_out, d_value_out);  // read full history
cache.reset();                                        // reuse pages for next request
```

---

### Sampler (`sampler.cpp`)

Converts raw logits `[vocab_size]` to a single sampled token ID.

**Sampling pipeline (per decoding step):**

1. **Temperature scaling** — `logits /= temperature`  
   (`temperature = 0` → greedy argmax)
2. **Top-k filter** — zero out all but the top-k logits
3. **Softmax** — convert logits to probabilities (uses the online softmax kernel)
4. **Top-p (nucleus) filter** — keep the smallest set with cumulative probability ≥ `top_p`, renormalize
5. **Sample** — draw one token from the filtered distribution

**RNG:** Uses a fast linear congruential generator (LCG) with no dependency on `std::mt19937` — fully deterministic given a fixed seed, with no hidden state leakage between calls.

---

## Python Bindings (`src/bindings/python.cpp`)

pybind11 bridge exposing the C++ engine to Python.

The binding layer:
1. Accepts NumPy arrays (via `py::array_t<float>`)
2. Validates shapes and dtypes
3. Calls the CUDA host launcher functions
4. Returns results as new NumPy arrays

The `NanoInfer` Python class wraps `GPT2Model` + `PagedKVCache` + `Sampler` behind the HuggingFace-compatible `from_pretrained` / `generate` interface.

---

## Build System

### Two build paths

**CMake** (preferred for C++ development):
- Produces static libraries: `nanoinfer_kernels` + `nanoinfer_engine`
- Produces benchmark executables and CUDA test executables
- Optionally builds Python bindings via `add_subdirectory(src/bindings)` if pybind11 is found

**setup.py** (preferred for Python packaging):
- Custom `NvccBuildExt` class compiles `.cu` files with `nvcc -O3 --use_fast_math`
- Links all CUDA objects into a single `nanoinfer.so` extension module
- Auto-detects CUDA home from `CUDA_HOME` env var, `nvcc` on PATH, or glob of `/usr/local/cuda-*`

### CUDA architecture targets

| `sm_` | GPU | Notes |
|---|---|---|
| `sm_80` | A100 (Ampere) | Primary development target |
| `sm_86` | RTX 3090, A6000 | Consumer Ampere |
| `sm_89` | RTX 4090 (Ada Lovelace) | `dp4a` available; FP8 not |
| `sm_90` | H100 (Hopper) | Required for FP8 (`__nv_fp8_e4m3`) — on roadmap |

Override from CMake: `cmake .. -DCUDA_ARCHITECTURES="80;86;90"`

---

## Performance Model

### Where the speedup comes from

| Optimization | Where applied | Speedup |
|---|---|---|
| Tiled shared memory GEMM | All matrix multiplications | ~4× vs naïve GEMM |
| Bank conflict elimination (padding) | GEMM, INT8 GEMM | Eliminates 16-way serialization |
| Online softmax (1-pass) | Softmax, attention | ~3× vs 3-pass safe softmax |
| Flash Attention tiling | Multi-head attention | 3× latency, 22× memory |
| Fused LayerNorm+Linear | Every transformer sub-layer | 1.4× per layer |
| INT8 GEMM with `dp4a` | Quantized mode | 2–3× vs FP32 GEMM |
| Paged KV-cache | Autoregressive generation | Avoids memory fragmentation |

### Bottleneck analysis

For long sequences, the dominant cost is **attention** (quadratic in sequence length). Flash Attention's O(N) memory scaling is the key enabler for sequences > 1024 tokens on consumer hardware.

For short sequences (e.g., single-token decode steps), the bottleneck shifts to **memory bandwidth** (loading weights from HBM). At this point, INT8 quantization's 4× weight size reduction directly translates to ~2–3× throughput improvement.

The remaining gap to TensorRT (2× at INT8) is primarily explained by:
- **Layer fusion graph optimization** — TensorRT fuses more ops at compile time
- **Kernel auto-tuning** — TensorRT profiles multiple kernel variants and selects the best for each shape
- **Epilogue fusion** — TensorRT fuses activation and quantization into GEMM epilogues

---

## Roadmap

| Feature | Status |
|---|---|
| Tiled GEMM (FP32, tile 16 & 32) | ✅ Done |
| Online Softmax with warp reduction | ✅ Done |
| Flash Attention v1 (warp-per-row) | ✅ Done |
| Fused LayerNorm + Linear | ✅ Done |
| GPT-2 full pipeline | ✅ Done |
| INT8 GEMM with `dp4a` | ✅ Done |
| Flash Attention v2 (warp partitioning) | 🔲 Planned |
| LLaMA-3 support | 🔲 Planned |
| Multi-GPU tensor parallel | 🔲 Planned |
| CUTLASS-backed GEMM comparison | 🔲 Planned |
| FP8 support (H100+, `sm_90`) | 🔲 Planned |
