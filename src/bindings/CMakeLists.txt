# CMakeLists.txt — bindings/
# Builds the pybind11 Python extension module: nanoinfer.so
#
# This file is included by the top-level CMakeLists.txt via:
#   add_subdirectory(src/bindings)
#
# The resulting .so is placed in the repo root so Python can import it
# without an explicit install step (useful during development):
#   import sys; sys.path.insert(0, "/path/to/nanoinfer")
#   import nanoinfer

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Find pybind11
# ---------------------------------------------------------------------------
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# Collect all source files that go into the extension
# ---------------------------------------------------------------------------
set(BINDING_SOURCES
    python.cpp                         # pybind11 entry point (this directory)
    ../engine/model.cpp
    ../engine/kvcache.cpp
    ../engine/sampler.cpp
    ../kernels/gemm.cu
    ../kernels/softmax.cu
    ../kernels/attention.cu
    ../kernels/quantize.cu
    ../layers/linear.cu
    ../layers/layernorm.cu
    ../layers/embedding.cu
)

# ---------------------------------------------------------------------------
# pybind11_add_module handles:
#   - correct Python include paths
#   - correct shared library suffix (.so on Linux, .pyd on Windows)
#   - stripping of debug symbols in Release mode
# ---------------------------------------------------------------------------
pybind11_add_module(nanoinfer MODULE ${BINDING_SOURCES})

# ---------------------------------------------------------------------------
# Compiler settings
# ---------------------------------------------------------------------------

# C++ standard for host code
target_compile_features(nanoinfer PRIVATE cxx_std_17)

# CUDA-specific flags (applied to .cu translation units only)
target_compile_options(nanoinfer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -std=c++17
        --compiler-options -fPIC
        # Target Ampere (A100 = sm_80) and Ampere consumer (RTX 3090 = sm_86).
        # Add -gencode arch=compute_90,code=sm_90 for H100 (FP8 support).
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
        -Wextra
    >
)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(nanoinfer PRIVATE
    ${CMAKE_SOURCE_DIR}/src        # allows #include "kernels/gemm.h" etc.
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ---------------------------------------------------------------------------
# Link against CUDA runtime and cuBLAS
# ---------------------------------------------------------------------------
target_link_libraries(nanoinfer PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# ---------------------------------------------------------------------------
# Output location: place the .so in the repo root for easy import
# ---------------------------------------------------------------------------
set_target_properties(nanoinfer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/nanoinfer
)

# ---------------------------------------------------------------------------
# Optional: install target (used by `pip install .` via setup.py)
# ---------------------------------------------------------------------------
install(TARGETS nanoinfer
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python
)