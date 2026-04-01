#pragma once
#include <cuda_runtime.h>

// Linear layer: output = input @ weight^T + bias
// input: [batch x in_features]
// weight: [out_features x in_features]
// bias:   [out_features] (nullable)
// output: [batch x out_features]
void launch_linear(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,    // pass nullptr to skip bias
    float*       d_output,
    int batch, int in_features, int out_features
);

// Fused LayerNorm + Linear — eliminates one HBM round-trip
// Normalizes input first, then applies linear projection in the same kernel
void launch_layernorm_linear_fused(
    const float* d_input,   // [batch x hidden]
    const float* d_gamma,   // [hidden]
    const float* d_beta,    // [hidden]
    const float* d_weight,  // [out_features x hidden]
    const float* d_bias,    // [out_features] (nullable)
    float*       d_output,  // [batch x out_features]
    int batch, int hidden, int out_features,
    float eps = 1e-5f
);