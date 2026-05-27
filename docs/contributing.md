# Contributing to NanoInfer

Thank you for your interest in contributing! NanoInfer is a learning project, so contributions that improve clarity, correctness, or performance are especially welcome.

---

## Development Setup

```bash
git clone https://github.com/lecuong1502/NanoInfer.git
cd NanoInfer

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"   # installs pytest, torch, transformers for tests
```

Verify your setup:

```bash
# Build and run all tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=80
make -j$(nproc)
ctest --test-dir . -V
cd ..
pytest tests/test_kernels.py -v
```

---

## Code Style

- **C++/CUDA:** Follow the existing style — 4-space indent, `snake_case` for functions and variables, comments explain *why* not *what*.
- **Python:** PEP 8. Type hints required for all public functions.
- **Comments in kernels:** Every non-obvious optimization must have an inline comment explaining the rationale (cycle counts, bank conflict fix, etc.).

---

## Adding a New Kernel

1. Create `src/kernels/my_kernel.cu` and `src/kernels/my_kernel.h`
2. Add a host launcher function `void launch_my_kernel(...)` in the `.cu` file
3. Add the `.cu` to both `CMakeLists.txt` (under `nanoinfer_kernels`) and `setup.py` (under `sources`)
4. Write a correctness test in `tests/test_my_kernel.cu` comparing against a reference (usually PyTorch)
5. Add a Python binding in `src/bindings/python.cpp` if the kernel should be accessible from Python
6. Update `nanoinfer/nanoinfer.pyi` with type stubs for the new Python-facing function
7. Document in `docs/architecture.md`

### Test template (CUDA)

```cuda
// tests/test_my_kernel.cu
#include "../src/kernels/my_kernel.h"
#include <cassert>
#include <cmath>

int main() {
    // 1. Allocate and initialize host data
    // 2. Copy to device
    // 3. Run kernel
    // 4. Copy result back to host
    // 5. Compare against reference with tolerance
    //    assert(fabs(got - expected) < 1e-4f);
    printf("PASSED\n");
    return 0;
}
```

---

## Adding Model Support

To add a new model architecture (e.g., LLaMA-3):

1. Define a new config struct in `src/engine/` (similar to `GPT2Config`)
2. Implement the forward pass by composing existing layer primitives
3. Add weight loading logic (from HuggingFace safetensors or a custom binary format)
4. Add a tokenizer (or reuse `tiktoken`/`sentencepiece` via Python)
5. Expose through the pybind11 layer in `src/bindings/python.cpp`

---

## Submitting a Pull Request

1. Fork the repository and create a feature branch
2. Make sure all existing tests pass: `ctest --test-dir build -V && pytest tests/ -v`
3. Add tests for your new feature
4. Write a clear PR description explaining the motivation, design decisions, and benchmark results (if performance-related)

---

## Reporting Bugs

Open a GitHub issue with:
- GPU model and CUDA version (`nvidia-smi`, `nvcc --version`)
- Python version and NanoInfer version (`python --version`, `pip show nanoinfer`)
- Minimal reproduction case
- Expected vs actual output

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](../LICENSE).
