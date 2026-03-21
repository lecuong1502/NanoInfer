#pragma once
#include <cuda_runtime.h>

// Flash Attention v1: IO-aware exact attention
// Q, K, V, O: [B x H x S x d_head]
void launch_flash_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int B, int N, int S, int d,
    bool causal = true
);