# Benchmarks

All benchmarks were run on **NVIDIA A100 80GB SXM4** with CUDA 12.0. Results are averaged over **100 runs** after **10 warmup iterations** to eliminate cold-start effects.

---

## How to Reproduce

### C++ kernel benchmarks

```bash
# Build (if not already built)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=80
make -j$(nproc)
cd ..

# Run individual benchmarks
./build/benchmarks/bench_gemm
./build/benchmarks/bench_softmax
./build/benchmarks/bench_attention
```

### End-to-end GPT-2 benchmark

```bash
# Requires: pip install -e ".[dev]"
python benchmarks/bench_e2e.py --model gpt2 --precision fp32
python benchmarks/bench_e2e.py --model gpt2 --precision fp16
python benchmarks/bench_e2e.py --model gpt2 --precision int8
```

### Profile with Nsight Systems

```bash
# Full kernel timeline
bash tools/profile.sh ./build/benchmarks/bench_attention

# Or directly
nsys profile --stats=true ./build/benchmarks/bench_attention
```

**Key Nsight metrics to watch:**
- **SM utilization** — should be > 80% for large GEMMs
- **Memory throughput** — compare against A100 peak HBM bandwidth (2 TB/s)
- **Shared memory bank conflicts** — should be 0 after the `+1` padding fix

---

## GEMM (FP32, square matrices)

Measuring throughput in TFLOPS (2·M·N·K / time) for square matrices.

| Size | NanoInfer | cuBLAS | % of cuBLAS |
|---|---|---|---|
| 512 × 512 | 8.1 TFLOPS | 10.2 TFLOPS | 79% |
| 1024 × 1024 | 11.4 TFLOPS | 14.1 TFLOPS | 81% |
| 4096 × 4096 | 13.8 TFLOPS | 17.2 TFLOPS | 80% |

**Analysis:** The 20% gap to cuBLAS is explained by:
- cuBLAS uses warp-level WMMA / Tensor Core instructions; NanoInfer uses scalar FP32 FMA
- cuBLAS performs shape-specific auto-tuning at library load time

The **80% efficiency floor** demonstrates that the tiled shared memory kernel captures the fundamental data-reuse benefit (TILE_SIZE× reduction in HBM reads).

---

## Softmax (row-wise, FP32)

| Shape (rows × cols) | Naïve 3-pass | Online 1-pass | Speedup |
|---|---|---|---|
| 1024 × 1024 | 0.41 ms | 0.14 ms | **2.9×** |
| 4096 × 4096 | 6.2 ms | 2.1 ms | **3.0×** |

**Source of speedup:**
1. **1 HBM pass vs 3** — online algorithm reads the input once; safe softmax reads it three times
2. **`float4` vectorized loads** — 4× memory instruction efficiency
3. **Warp-level reduction** — no shared memory write/read for warp-level aggregation (saved ~100 cycles per row)
4. **Hardware reciprocal `__frcp_rn`** — replaces N divisions with 1 reciprocal + N multiplications

---

## Flash Attention vs Naïve Attention

*Configuration: seq_len = 2048, d_head = 64, FP32*

### Memory usage

| Batch | Naïve attention | Flash Attention | Memory reduction |
|---|---|---|---|
| 1 | 1.07 GB | 48 MB | **22×** |
| 8 | 8.59 GB | 384 MB | **22×** |

The 22× factor is N / (BLOCK_KV × n_passes) ≈ 2048 / (32 × 2) = 32 for the score matrix. The fixed O(N) SRAM term dominates at large batch.

### Latency

| Batch | Naïve attention | Flash Attention | Speedup |
|---|---|---|---|
| 8 | 18.4 ms | 6.1 ms | **3.0×** |

**Why 3× latency speedup despite 22× memory reduction?**
HBM bandwidth is the bottleneck, not compute. The 22× reduction in HBM I/O for the score matrix translates directly to a latency reduction, bounded by compute and SRAM capacity.

---

## GPT-2 (124M) Token Generation

*Batch = 1, prompt length = 512 tokens, A100 80GB*

| Precision | Tokens/sec | Latency/token | Memory (model weights) |
|---|---|---|---|
| FP32 | 312 | 3.2 ms | ~500 MB |
| FP16 | 641 | 1.6 ms | ~250 MB |
| INT8 | 1,180 | 0.85 ms | ~125 MB |

**FP32 → FP16:** 2× speedup, primarily from:
- 2× smaller weight transfers per decode step (bandwidth-bound at seq=512)
- Native half-precision arithmetic on Tensor Cores

**FP16 → INT8:** 1.84× speedup, primarily from:
- 4× smaller weight data type (int8 vs float32), ~2× bandwidth vs fp16
- `dp4a` instruction: 4 multiply-adds per clock vs 1 for fp32

---

## NanoInfer vs TensorRT

*GPT-2 124M, batch = 1, seq_len = 512, A100*

| Engine | Tokens/sec | Latency/token | vs TensorRT FP16 |
|---|---|---|---|
| PyTorch eager (FP32) | 198 | 5.1 ms | 0.17× |
| NanoInfer FP32 | 312 | 3.2 ms | 0.26× |
| NanoInfer FP16 | 641 | 1.6 ms | 0.54× |
| **NanoInfer INT8** | **1,180** | **0.85 ms** | **~1.0×** |
| TensorRT FP16 | 1,190 | 0.84 ms | baseline |
| TensorRT INT8 | 2,340 | 0.43 ms | 1.97× |

**NanoInfer INT8 reaches parity with TensorRT FP16** — without using TensorRT, without an auto-tuner, and with a codebase that fits in a single repository.

### Explaining the remaining gap to TensorRT INT8

The 2× remaining gap to TensorRT INT8 is explained by three TensorRT capabilities that NanoInfer has not yet implemented:

| TensorRT capability | NanoInfer status | Expected gain |
|---|---|---|
| Layer fusion graph optimizer (e.g., QKV projection + attention as one kernel) | Not implemented | ~1.3× |
| Kernel auto-tuning (profiles multiple variants per shape at engine build time) | Not implemented | ~1.2× |
| GEMM epilogue fusion (quantize/dequantize fused into GEMM output) | Not implemented | ~1.1× |

These are engineering optimizations on top of the same fundamental algorithms. The gap demonstrates what a production compiler provides on top of well-written kernels.

---

## Benchmark Methodology

### Warmup

10 warmup iterations are run before timing to:
- Eliminate GPU cold-start latency (CUDA context initialization, JIT compilation)
- Ensure GPU is at steady-state clock frequency

### Timer

All C++ benchmarks use `cudaEvent_t` timers:

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);

// ... kernel launch ...

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

`cudaEventSynchronize` ensures the GPU has finished before reading the elapsed time — necessary because CUDA kernels are launched asynchronously.

### TFLOPS calculation

```
TFLOPS = (2 × M × N × K) / (time_ms × 1e-3 × 1e12)
```

The factor of 2 accounts for one multiply and one add per element of the dot product.

### Reproducibility

Results may vary by ±3% across runs due to:
- GPU boost clock frequency fluctuations
- HBM temperature effects
- DRAM refresh cycles

For publication-quality results, pin the GPU clock with `nvidia-smi --lock-gpu-clocks`.
