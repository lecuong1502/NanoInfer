# Getting Started

This guide walks you through installing NanoInfer, building from source, and running your first inference.

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| CUDA Toolkit | 12.0 | Must include `nvcc` on `PATH` |
| NVIDIA GPU | Compute Capability ≥ 8.0 | Ampere (A100, RTX 3080) or newer |
| CMake | 3.20 | For native C++ / benchmark builds |
| Python | 3.9 | For bindings, tests, and end-to-end benchmarks |
| pybind11 | 2.11 | Installed automatically via pip |
| numpy | 1.23 | Installed automatically via pip |

**Optional (for benchmarking and profiling):**
- cuBLAS — ships with the CUDA Toolkit, used as a comparison baseline
- NVIDIA Nsight Systems — for kernel timeline profiling

---

## Installation

### Option A — pip (recommended)

The easiest path. `setup.py` auto-detects your CUDA installation and compiles everything.

```bash
git clone https://github.com/lecuong1502/NanoInfer.git
cd NanoInfer

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

After installation, `import nanoinfer` is available from anywhere inside the virtualenv.

> **Tip:** If `nvcc` is not on your `PATH`, set `CUDA_HOME` before running pip:
> ```bash
> CUDA_HOME=/usr/local/cuda-12.0 pip install -e .
> ```

### Option B — CMake (for C++ benchmarks and tests)

Use this path if you want to run the C++ benchmark executables or native CUDA tests.

```bash
git clone https://github.com/lecuong1502/NanoInfer.git
cd NanoInfer

mkdir build && cd build

# Target A100 (sm_80). Change CUDA_ARCHITECTURES for your GPU:
#   RTX 3090 / A6000  →  86
#   RTX 4090           →  89
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=80

make -j$(nproc)
```

Built artifacts:
| Path | Description |
|---|---|
| `build/benchmarks/bench_gemm` | GEMM vs cuBLAS benchmark |
| `build/benchmarks/bench_softmax` | Softmax variants benchmark |
| `build/benchmarks/bench_attention` | Flash Attention vs naive benchmark |
| `build/tests/test_gemm` | GEMM correctness test |
| `build/tests/test_softmax` | Softmax correctness test |
| `build/tests/test_attention` | Attention correctness test |
| `build/tests/test_quantize` | INT8 quantization correctness test |

---

## Quick Start

### 1. Generate text with GPT-2

```python
from nanoinfer import NanoInfer

# Load GPT-2 (124M) with INT8 quantization for maximum throughput
model = NanoInfer.from_pretrained("gpt2", precision="int8")

output = model.generate(
    prompt="The quick brown fox",
    max_tokens=50,
    temperature=0.8,
    top_p=0.95,
)
print(output)
```

### 2. Use low-level kernels directly

```python
import numpy as np
import nanoinfer.kernels as kernels

# Tiled GEMM
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = kernels.gemm(A, B)          # shape (1024, 1024)

# Online softmax (single-pass, numerically stable)
x = np.random.randn(512, 4096).astype(np.float32)
y = kernels.softmax(x)          # each row sums to 1.0

# Flash Attention
Q = np.random.randn(1, 12, 512, 64).astype(np.float32)  # (batch, heads, seq, d_head)
K = np.random.randn(1, 12, 512, 64).astype(np.float32)
V = np.random.randn(1, 12, 512, 64).astype(np.float32)
O = kernels.flash_attention(Q, K, V, causal=True)
```

### 3. Run the benchmark suite

```bash
# C++ kernel benchmarks
./build/benchmarks/bench_gemm
./build/benchmarks/bench_softmax
./build/benchmarks/bench_attention

# End-to-end GPT-2 benchmark (Python)
python benchmarks/bench_e2e.py --model gpt2 --precision int8

# Profile with Nsight Systems
bash tools/profile.sh ./build/benchmarks/bench_attention
```

### 4. Run tests

```bash
# Native CUDA tests (via CTest)
ctest --test-dir build -V

# Or run individually
./build/tests/test_gemm
./build/tests/test_softmax
./build/tests/test_attention
./build/tests/test_quantize

# Python tests
pytest tests/test_kernels.py -v
```

All correctness tests compare output against PyTorch reference values with tolerance `atol=1e-4, rtol=1e-4`.

---

## Choosing a Precision

| Precision | Flag | Tokens/sec (GPT-2, A100) | Use when… |
|---|---|---|---|
| `fp32` | `precision="fp32"` | 312 | Debugging; maximum numerical accuracy |
| `fp16` | `precision="fp16"` | 641 | Production on Ampere+ GPUs |
| `int8` | `precision="int8"` | 1,180 | Maximum throughput; matches TensorRT FP16 |

---

## Next Steps

- **[Architecture](architecture.md)** — deep dive into every kernel and design decision
- **[API Reference](api-reference.md)** — complete Python API documentation
- **[Benchmarks](benchmarks.md)** — methodology, numbers, and how to reproduce them
- **[Contributing](../CONTRIBUTING.md)** — how to add new kernels or model support
