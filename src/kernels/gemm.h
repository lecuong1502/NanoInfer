#pragma once
#include <cuda_runtime.h>

// Tiled GEMM: C = A @ B
// A: [M x K], B: [K x N], C: [M x N]
void launch_gemm_tiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K
);