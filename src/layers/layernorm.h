#pragma once
#include <cuda_runtime.h>

// Standard LayerNorm
// output[i] = (input[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
// input/output: [rows x cols]
// gemma, beta: [cols]
void launch_layernorm(
    const float* d_input,
    const float* d_gamma,
    const float* d_beta,
    float* d_output,
    int rows, int cols,
    float eps = 1e-5f
);