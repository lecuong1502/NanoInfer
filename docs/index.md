# NanoInfer Documentation

**NanoInfer** is a lightweight, from-scratch CUDA inference engine for transformer models. No PyTorch. No TensorRT. Just raw CUDA kernels, hand-optimized for performance.

---

## What is NanoInfer?

NanoInfer is a GPU inference engine built to make the internals of high-performance inference *legible*. Every kernel is written by hand. Every optimization is explicit and benchmarked. The goal is to understand *why* systems like TensorRT are fast — by building the same techniques from scratch.

It covers the same ground as an NVIDIA inference team works on daily:

- **Tiled GEMM** with shared memory — reaching ~80% of cuBLAS peak TFLOPS
- **Online Softmax** — single-pass numerically stable, 3× faster than the 3-pass version
- **Flash Attention v1** — O(N) memory, 3× latency speedup, 22× memory reduction
- **INT8 GEMM** with `dp4a` — 2–3× throughput over FP32, matching TensorRT FP16
- **Fused LayerNorm + Linear** — eliminates HBM round-trips between kernels (1.4× speedup)
- **Paged KV-cache** — memory-efficient autoregressive generation
- **GPT-2 full pipeline** — autoregressive inference with Python bindings

---

## Documentation

| Page | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation, build instructions, quick start examples |
| [API Reference](api-reference.md) | Complete Python API documentation |
| [Architecture](architecture.md) | Deep dive into every kernel and design decision |
| [Benchmarks](benchmarks.md) | Benchmark results, methodology, and how to reproduce |
| [Contributing](contributing.md) | How to add new kernels, models, or fixes |

---

## Quick Install

```bash
git clone https://github.com/lecuong1502/NanoInfer.git
cd NanoInfer
pip install -e ".[dev]"
```

**Requirements:** CUDA 12.0+, GPU with Compute Capability ≥ 8.0, Python 3.9+

---

## Quick Example

```python
from nanoinfer import NanoInfer

model = NanoInfer.from_pretrained("gpt2", precision="int8")
print(model.generate("The quick brown fox", max_tokens=50))
```

---

## Benchmark Highlights

| Engine | Tokens/sec | vs TensorRT FP16 |
|---|---|---|
| PyTorch eager FP32 | 198 | 0.17× |
| NanoInfer FP32 | 312 | 0.26× |
| NanoInfer FP16 | 641 | 0.54× |
| **NanoInfer INT8** | **1,180** | **~1.0×** |
| TensorRT FP16 | 1,190 | baseline |

*GPT-2 124M, batch=1, seq=512, NVIDIA A100 80GB*

---

## License

[MIT](../LICENSE) — © lecuong1502
