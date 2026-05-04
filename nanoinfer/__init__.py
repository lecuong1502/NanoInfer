"""
NanoInfer — lightweight CUDA inference engine for transformer models.

Public API
----------
High-level (HuggingFace-compatible):
    from nanoinfer import NanoInfer
    model = NanoInfer.from_pretrained("gpt2", precision="int8")
    print(model.generate("The quick brown fox", max_tokens=50))

Low-level kernel access (for benchmarking / research):
    import nanoinfer.kernels as kernels
    C = kernels.gemm(A, B)                         # tiled GEMM
    y = kernels.softmax(x)                         # online softmax
    O = kernels.flash_attention(Q, K, V, causal=True)
"""

from importlib import import_module as _import_module
import sys as _sys

# ---------------------------------------------------------------------------
# Lazy-load the compiled C extension (nanoinfer.so / nanoinfer.pyd).
# This gives a clean ImportError with a helpful message if the .so is missing.
# ---------------------------------------------------------------------------

def _load_extension():
    try:
        # The C extension is compiled to nanoinfer/nanoinfer.so (via CMake)
        # or placed at the package root (via setup.py build_ext --inplace).
        # Either way, Python finds it as a sub-module of this package.
        from . import nanoinfer as _ext  # noqa: F401
        return _ext
    except ImportError:
        raise ImportError(
            "NanoInfer C extension not found.\n\n"
            "Build it with one of:\n"
            "  pip install -e .                         (editable install)\n"
            "  pip install -e '.[dev]'                  (+ test deps)\n"
            "  python setup.py build_ext --inplace      (in-place, no install)\n\n"
            "Requirements: CUDA 12+, pybind11, CMake 3.20+\n"
            "  pip install pybind11\n"
        ) from None


_ext = _load_extension()

# ---------------------------------------------------------------------------
# Re-export the public API at the top level
# ---------------------------------------------------------------------------

#: High-level GPT-2 inference class (HuggingFace-compatible API)
NanoInfer = _ext.NanoInfer

#: Low-level kernel sub-module (gemm, softmax, flash_attention)
kernels = _ext.kernels

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__   = "0.1.0"
__author__    = "lecuong1502"
__license__   = "MIT"
__all__       = ["NanoInfer", "kernels", "__version__"]
