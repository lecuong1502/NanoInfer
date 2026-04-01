#pragma once
#include <cuda_runtime.h>

// Element residual add: dst += src (in-place)
// dst, src: [n_elements]
void launch_add(
    float*       d_dst,
    const float* d_src,
    int          n_elements);

// GELU activation (in-place): x = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
// GPT-2 uses the tanh approximation of GELU (not the exact erf version)
// x: [n_elements]
void launch_gelu(
    float* d_x,
    int n_elements
);