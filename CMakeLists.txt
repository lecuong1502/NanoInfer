cmake_minimum_required(VERSION 3.20)

project(nanoinfer
    VERSION     0.1.0
    DESCRIPTION "Lightweight CUDA inference engine for transformer models"
    LANGUAGES   CXX CUDA
)

# ---------------------------------------------------------------------------
# Global standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build type (default: Release)
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---------------------------------------------------------------------------
# CUDA architecture targets
# ---------------------------------------------------------------------------
# sm_80 = A100 (Ampere)
# sm_86 = RTX 3090 / A6000 (Ampere consumer)
# sm_89 = RTX 4090 (Ada Lovelace)
# sm_90 = H100 (Hopper) — needed for FP8 support
#
# Override from command line: cmake .. -DCUDA_ARCHITECTURES="80;86;90"
if(NOT DEFINED CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES "80;86" CACHE STRING "CUDA SM architectures to target")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
set(NANOINFER_CXX_FLAGS  -O3 -Wall -Wextra)
set(NANOINFER_CUDA_FLAGS
    -O3
    --use_fast_math
    --compiler-options -fPIC
)

# ---------------------------------------------------------------------------
# Find required packages
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_DIR}")
 
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found — Python bindings will be built")
else()
    message(WARNING "pybind11 not found — skipping Python bindings (src/bindings/)")
endif()

# ---------------------------------------------------------------------------
# Shared include directory (every sub-target adds this)
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# Core CUDA kernel library
# Compiled once, linked into benchmarks, tests, and the Python extension.
# ---------------------------------------------------------------------------
add_library(nanoinfer_kernels STATIC
    src/kernels/gemm.cu
    src/kernels/softmax.cu
    src/kernels/attention.cu
    src/kernels/quantize.cu
    src/layers/linear.cu
    src/layers/layernorm.cu
    src/layers/embedding.cu
)

target_compile_options(nanoinfer_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NANOINFER_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${NANOINFER_CXX_FLAGS}>
)

target_link_libraries(nanoinfer_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(nanoinfer_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON   # allows __device__ symbols across .cu files
    POSITION_INDEPENDENT_CODE  ON   # required when linking into a shared .so
)

# ---------------------------------------------------------------------------
# Engine library (C++ only — model assembly, KV-cache, sampler)
# ---------------------------------------------------------------------------
add_library(nanoinfer_engine STATIC
    src/engine/model.cpp
    src/engine/kvcache.cpp
    src/engine/sampler.cpp
)

target_compile_options(nanoinfer_engine PRIVATE ${NANOINFER_CXX_FLAGS})
 
target_link_libraries(nanoinfer_engine PUBLIC
    nanoinfer_kernels
)
 
set_target_properties(nanoinfer_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
add_executable(bench_gemm benchmarks/bench_gemm.cu)
add_executable(bench_softmax benchmarks/bench_softmax.cu)
add_executable(bench_attention benchmarks/bench_attention.cu)
 
foreach(bench bench_gemm bench_softmax bench_attention)
    target_compile_options(${bench} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${NANOINFER_CUDA_FLAGS}>
    )
    target_link_libraries(${bench} PRIVATE nanoinfer_kernels)
    set_target_properties(${bench} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
    )
endforeach()

# ---------------------------------------------------------------------------
# Python bindings (optional — only if pybind11 is available)
# ---------------------------------------------------------------------------
if(pybind11_FOUND)
    add_subdirectory(src/bindings)
endif()

# ---------------------------------------------------------------------------
# Install targets
# ---------------------------------------------------------------------------
install(TARGETS nanoinfer_kernels nanoinfer_engine
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/nanoinfer
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)
 
# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========== NanoInfer build summary ==========")
message(STATUS "  Version           : ${PROJECT_VERSION}")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA version      : ${CUDAToolkit_VERSION}")
message(STATUS "  Python bindings   : ${pybind11_FOUND}")
message(STATUS "  .so output        : ${CMAKE_SOURCE_DIR}/nanoinfer/")
message(STATUS "=============================================")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=80")
message(STATUS "  make -j\$(nproc)")
message(STATUS "")