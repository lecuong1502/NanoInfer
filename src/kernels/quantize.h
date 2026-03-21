#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// INT8 GEMM using dp4a instruction
// A: [M x K] int8, B: [K x N] int8, C: [M x N] int32
void launch_int8_gemm(
    const int8_t* d_A,
    const int8_t* d_B,
    int32_t* d_C,
    int M, int N, int K
);

// Quantize float32 -> int8 with per-tensor scale
void launch_quantize(
    const float* d_input,
    int8_t* d_output,
    float scale,
    int n_elements
);

// Dequantize int32 accumulator -> float32
void launch_dequantize(
    const int32_t* d_input,
    float* d_output,
    float scale_A,
    float scale_B,
    int n_elements
);