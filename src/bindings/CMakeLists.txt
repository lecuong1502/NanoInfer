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
    ../layers/activation.cu
    ../layers/linear.cu
    ../layers/layernorm.cu
    ../layers/embedding.cu
)

# ---------------------------------------------------------------------------
# pybind11_add_module handles:
#   - correct Python include paths
#   - correct shared library suffix (.so on Linux, .pyd on Windows)
#   - stripping of debug symbols in Release mode
#
# NO_EXTRAS: disable pybind11's automatic -flto=auto injection.
#   pybind11 3.x adds -flto=auto to ALL languages including CUDA.
#   nvcc does not understand -flto=auto and treats it as an input file,
#   causing: "A single input file is required for a non-link phase".
#   We re-add -fvisibility=hidden manually for C++ only (see below).
# ---------------------------------------------------------------------------
pybind11_add_module(nanoinfer MODULE NO_EXTRAS ${BINDING_SOURCES})

# -fvisibility=hidden: required for correct Python extension symbol export
# Scoped to CXX only — do NOT pass to CUDA (nvcc handles visibility differently)
target_compile_options(nanoinfer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)

# Disable CUDA separable compilation for the pybind11 module.
# The .cu files here are standalone — no cross-file __device__ symbols.
set_target_properties(nanoinfer PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

# ---------------------------------------------------------------------------
# Compiler settings
# ---------------------------------------------------------------------------

# Build command:
# mkdir -p build && cd build && \
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=89 \
#   -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") && \
#   make nanoinfer -j$(nproc)

# C++ standard for host code
target_compile_features(nanoinfer PRIVATE cxx_std_17)

# CUDA-specific flags (applied to .cu translation units only)
set(CUDA_FLAGS
    "-O3"
    "--use_fast_math"
)

set(CXX_FLAGS
    "-O3"
    "-Wall"
    "-Wextra"
)

# Truyền list flags vào generator expressions
target_compile_options(nanoinfer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
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