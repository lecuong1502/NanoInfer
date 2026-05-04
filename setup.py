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
    # 1. Explicit env variable — highest priority
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and Path(cuda_home).exists():
        return cuda_home

    # 2. Derive from nvcc on PATH: nvcc lives at <cuda_home>/bin/nvcc
    try:
        nvcc = subprocess.check_output(
            ["which", "nvcc"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return str(Path(nvcc).resolve().parent.parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3. Glob versioned installs, e.g. /usr/local/cuda-12.9, prefer newest
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

    NVCC_ARCH_FLAGS = [
        "-gencode", "arch=compute_80,code=sm_80",   # A100 (Ampere)
        "-gencode", "arch=compute_86,code=sm_86",   # RTX 3090 / A6000
        "-gencode", "arch=compute_89,code=sm_89",   # RTX 4050/4090 (Ada Lovelace)
    ]

    NVCC_FLAGS = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--compiler-options", "-fPIC",
        *NVCC_ARCH_FLAGS,
    ]

    def build_extension(self, ext):
        all_sources   = list(ext.sources)
        cuda_sources  = [s for s in all_sources if s.endswith(".cu")]
        other_sources = [s for s in all_sources if not s.endswith(".cu")]

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        cuda_objects = []
        for src in cuda_sources:
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

        ext.sources       = other_sources
        ext.extra_objects = list(ext.extra_objects or []) + cuda_objects
        super().build_extension(ext)


# ---------------------------------------------------------------------------
# pybind11 include path
#
# 3-method detection để hoạt động trong mọi trường hợp:
#   Method 1 — import trực tiếp (hoạt động khi venv đã activate)
#   Method 2 — chạy sys.executable trong subprocess
#              (fix khi pip chạy setup.py trong subprocess không có venv)
#   Method 3 — glob tìm trong .venv/ cạnh repo (fallback cuối cùng)
# ---------------------------------------------------------------------------

def find_pybind11_include() -> str:
    # Method 1: import trực tiếp
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        pass

    # Method 2: dùng python của venv hiện tại
    try:
        out = subprocess.check_output(
            [sys.executable, "-c",
             "import pybind11; print(pybind11.get_include())"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: glob tìm trong venv cạnh repo
    repo_root = Path(__file__).parent
    for venv_name in [".venv", "venv", "env"]:
        hits = list(
            (repo_root / venv_name / "lib").glob(
                "python*/site-packages/pybind11/include"
            )
        )
        if hits:
            return str(hits[0])

    raise RuntimeError(
        "pybind11 not found.\n"
        "  Activate your venv:    source .venv/bin/activate\n"
        "  Install pybind11:      pip install pybind11\n"
        "  Then retry:            pip install -e '.[dev]'"
    )


PYBIND11_INCLUDE = find_pybind11_include()
print(f"[setup.py] pybind11     = {PYBIND11_INCLUDE}")


# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------

SRC_ROOT = Path(__file__).parent / "src"

nanoinfer_ext = Extension(
    name="nanoinfer",
    sources=[
        "src/bindings/python.cpp",
        "src/engine/model.cpp",
        "src/engine/kvcache.cpp",
        "src/engine/sampler.cpp",
        "src/kernels/gemm.cu",
        "src/kernels/softmax.cu",
        "src/kernels/attention.cu",
        "src/kernels/quantize.cu",
        "src/layers/activation.cu",
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
    extra_objects=[],
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