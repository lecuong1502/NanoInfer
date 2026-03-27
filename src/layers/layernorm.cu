#include "layernorm.h"

// --------------------------------------------------------
// Layer Normalization
//
// Formula:
//  mean = sum(x) / H
//  var = sum((x - mean)^2) / H
//  x_hat = (x - mean) / sqrt(var + eps)
//  y = gamma * x_hat + beta
//
// In Transformer, LayerNorm appears 2 times per layer:
//  1. Before attention (Pre-LN) or after attention (Post-LN)
//  2. Before FFN (Pre-LN) or after FFN (Post-LN)
//
// GPT-2 uses Pre-LN — normalize before sub-layer
//
// Why need eps: avoid to divide 0 when all same values (var = 0)
// Value eps = 1e-5 is convention PyTorch LayerNorm
// -------------------------------------------------------


// Warp-level reduction (share for both mean and variance)
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Unroll completely — compiler knows the number of iterations (5 times)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Block-level reduction through shared memory
// Return result in all threads (broadcast to 0, then sync)
__device__ float block_reduce_sum_broadcast(float v, float* smem_scratch) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_sum(v);

    if (lane == 0) {
        smem_scratch[warp_id] = v;
    }
    __syncthreads();

    // First warp reduces warp partial sums
    int num_warps = (blockDim.x + 31) / 32;
    v = (threadIdx.x < num_warps) ? smem_scratch[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        v = warp_reduce_sum(v);
    }
    // Broadcast all threads through shared memory
    if (threadIdx.x == 0) {
        smem_scratch[0] = v;
    }
    __syncthreads();
    return smem_scratch[0];
}

// --------------------------------------------------
// LayerNorm kernel
//
// Each block handles 1 row — compute mean and var through 2 passes
// Mean and var need all rows to compute → It is not possible to tile in the direction of the columns.
// --------------------------------------------------
__global__ void layernorm_kernel(
    const float* __restrict__ input,   // [rows x cols]
    const float* __restrict__ gamma,   // [cols]
    const float* __restrict__ beta,    // [cols]
    float* __restrict__ output,  // [rows x cols]
    int rows, int cols,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input  + row * cols;
    float* y = output + row * cols;

    // Shared memory scratchpad for block reduction
    // Up to 32 warps per block → need 32 floats
    __shared__ float smem[32];

    // Pass 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += x[i];
    }
    float mean = block_reduce_sum_broadcast(sum, smem) / cols;

    // Pass 2: Compute variance
    // Use Welford's algorithm to compute var in 1 pass
    // Because we had mean, used naive 2-pass still stable
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float var    = block_reduce_sum_broadcast(var_sum, smem) / cols;
    float inv_std = rsqrtf(var + eps);

    // Apply normalize + affine transform
    // Gamma and Beta are learnable parameters, read once from HBM
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ---------------------------------------------
// Host launcher
// ---------------------------------------------
void launch_layernorm(
    const float* d_input,
    const float* d_gamma,
    const float* d_beta,
    float*       d_output,
    int rows, int cols,
    float eps
) {
    // 1 block per row, 256 threads per block
    // cols > 256: each thread handles multiple elements through stride loop
    // cols <= 256: Unnecessary threads that do nothing (idle) — acceptable
    int threads = min(256, cols);
    layernorm_kernel<<<rows, threads>>>(
        d_input, d_gamma, d_beta, d_output, rows, cols, eps
    );
}