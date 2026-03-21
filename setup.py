"""
setup.py — build NanoInfer Python bindings via pybind11 + nvcc.
 
Usage:
    pip install -e .          # editable install (development)
    pip install .             # regular install
    python setup.py build_ext --inplace   # build .so in-place without installing
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# ---------------------------------------------------------------------------
# Detect CUDA toolkit location
# Handles both standard layout (/usr/local/cuda/include) and
# non-standard layout (/usr/local/cuda-12.9/targets/x86_64-linux/include)
# ---------------------------------------------------------------------------

def find_cuda_home():
    # Explicit env variable — highest priority
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and Path(cuda_home).exists():
        return cuda_home
    
    # Derive from nvcc on PATH: nvcc lives at <cuda_home>/bin/nvcc
    try:
        nvcc = subprocess.check_output(
            ["which", "nvcc"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return str(Path(nvcc).resolve().parent.parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Glob versioned installs, e.g. /usr/local/cuda-12.9, prefer newest
    candidates = sorted(
        glob.glob("/usr/local/cuda-*") + ["/usr/local/cuda", "/usr/cuda"],
        reverse=True,
    )
    for c in candidates:
        if Path(c).exists():
            return c
        
    raise RuntimeError(
        "CUDA toolkit not found. "
        "Set CUDA_HOME=/usr/local/cuda-12.9 before running setup.py."
    )


CUDA_HOME = find_cuda_home()

# Resolve include path — check standard layout first, then targets/ layout
_std_inc    = Path(CUDA_HOME) / "include"
_target_inc = Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include"
CUDA_INCLUDE = str(_std_inc if _std_inc.exists() else _target_inc)
 
# Resolve lib path — same approach
_std_lib    = Path(CUDA_HOME) / "lib64"
_target_lib = Path(CUDA_HOME) / "targets" / "x86_64-linux" / "lib"
CUDA_LIB    = str(_std_lib if _std_lib.exists() else _target_lib)
 
print(f"[setup.py] CUDA_HOME    = {CUDA_HOME}")
print(f"[setup.py] CUDA_INCLUDE = {CUDA_INCLUDE}")
print(f"[setup.py] CUDA_LIB     = {CUDA_LIB}")


# ---------------------------------------------------------------------------
# Custom build_ext that compiles .cu files with nvcc
# ---------------------------------------------------------------------------

class NvccBuildExt(build_ext):
    """Compile CUDA source files with nvcc, everything else with the default compiler."""
 
    # GPU architectures to target. Add sm_90 for H100 (FP8 support).
    NVCC_ARCH_FLAGS = [
        "-gencode", "arch=compute_80,code=sm_80",   # A100
        "-gencode", "arch=compute_86,code=sm_86",   # RTX 3090 / A6000
    ]

    NVCC_FLAGS = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--compiler-options", "-fPIC",
        *NVCC_ARCH_FLAGS,
    ]

    def build_extension(self, ext):
        # Separate source files by type
        cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
        other_sources = [s for s in ext.sources if not s.endswith(".cu")]
 
        # Compile each .cu file to a .o object with nvcc
        cuda_objects = []
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
 
        for src in cuda_sources:
            # Use full relative path in stem to avoid name collisions
            # e.g. src/kernels/gemm.cu -> kernels_gemm.cu.o
            stem = Path(src).stem
            obj  = str(build_temp / f"{stem}.cu.o")
            cmd  = [
                "nvcc",
                *self.NVCC_FLAGS,
                "-I", CUDA_INCLUDE,
                *[f"-I{inc}" for inc in (ext.include_dirs or [])],
                "-c", src,
                "-o", obj,
            ]
            print("nvcc:", " ".join(cmd))
            subprocess.check_call(cmd)
            cuda_objects.append(obj)
 
        # Replace sources with cpp-only list and append compiled cuda objects
        # Use a copy so we don't permanently mutate the Extension descriptor
        ext.sources      = other_sources
        ext.extra_objects = list(ext.extra_objects or []) + cuda_objects
        super().build_extension(ext)


# ---------------------------------------------------------------------------
# pybind11 include path
# ---------------------------------------------------------------------------

try:
    import pybind11
    PYBIND11_INCLUDE = pybind11.get_include()
except ImportError:
    raise RuntimeError("pybind11 not found. Install it: pip install pybind11")


# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------

SRC_ROOT = Path(__file__).parent / "src"

nanoinfer_ext = Extension(
    name="nanoinfer",
    sources=[
        # pybind11 entry point
        "src/bindings/python.cpp",
        # Engine (C++ only)
        "src/engine/model.cpp",
        "src/engine/kvcache.cpp",
        "src/engine/sampler.cpp",
        # CUDA kernels
        "src/kernels/gemm.cu",
        "src/kernels/softmax.cu",
        "src/kernels/attention.cu",
        "src/kernels/quantize.cu",
        # Layers
        "src/layers/linear.cu",
        "src/layers/layernorm.cu",
        "src/layers/embedding.cu",
    ],
    include_dirs=[
        PYBIND11_INCLUDE,
        CUDA_INCLUDE,
        str(SRC_ROOT),
    ],
    library_dirs=[CUDA_LIB],
    libraries=["cudart", "cublas"],
    extra_compile_args=["-O3", "-std=c++17"],
    extra_objects=[],   # explicitly initialised — NvccBuildExt appends to this
    language="c++",
)

# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    name="nanoinfer",
    version="0.1.0",
    author="Your Name",
    description="Lightweight CUDA inference engine for transformer models",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pybind11>=2.11",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "torch>=2.0",
            "transformers>=4.35",
        ]
    },
    ext_modules=[nanoinfer_ext],
    cmdclass={"build_ext": NvccBuildExt},
    zip_safe=False,
)