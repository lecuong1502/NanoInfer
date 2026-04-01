#include "activation.h"

// ---------------------------------------------------
// Residual add: dst += src
//
// Appear twice per transformer block:
//      hidden += attn_out  (after attention)
//      hidden += ffn_out   (sau FFN)
//
// Use float4 vectorized load/store to increase memory throughput 4x
// ---------------------------------------------------
__global__ void add_kernel (
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 + 3 < n) {
        float4 d = reinterpret_cast<float4*>(dst)[i4 / 4];
        float4 s = reinterpret_cast<const float4*>(src)[i4 / 4];
        d.x += s.x; d.y += s.y; d.z += s.z; d.w += s.w;
        reinterpret_cast<float4*>(dst)[i4 / 4] = d;
    } else {
        // tail: handle remainder when n % 4 != 0
        for (int i = i4; i < n; i++)
            dst[i] += src[i];
    }
}

void launch_add(float* d_dst, const float* d_src, int n) {
    int threads = 256;
    int blocks  = (n / 4 + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_dst, d_src, n);
}


// -------------------------------------------------------------
// GELU activation (tanh approximation) — in-place
//
// GPT-2 use tanh GELU, not erf GELU:
//      GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// Why:
//  - erf() doesn't have hardware instructions on GPU -> slower
//  - tanh approximation error < 0.001 than exact GELU
//  - GPT-2 paper use tanh version, weights are trained by this version
//      -> Use erf when inference will return wrong result
//
// Constants:
//  sqrt(2/π) ≈ 0.7978845608
//  0.044715 is a coefficient in the polynomial approximation (Hendrycks & Gimpel 2016)
// ------------------------------------------------------------
__global__ void gelu_kernel(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = x[i];
    // tanh approximation: GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.044715x³)))
    float cube = v * v * v;
    float inner = 0.7978845608f * (v + 0.044715f * cube);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
    // tanhf: hardware intrinsic on CUDA, ~4 clock cycles
}

void launch_gelu(float* d_x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_x, n);
}