#include "attention.h"
#include <float.h>

// ---------------------------------------------------------------------------
// Flash Attention v1
//
// Problem with Naive attention:
//   S = Q @ K^T              → [B, H, N, N]  — N^2 memory!
//   P = softmax(S / sqrt(d)) → [B, H, N, N]
//   O = P @ V                → [B, H, N, d]
//
// With N=2048, d=64, batch=8, heads=12:
//   N*N = 4M floats per head → 4M * 8 * 12 * 4 bytes = 1.5 GB only for score matrix
//
// Flash Attention instead of materialize all NxN to HBM:
//  - Separate Q into Q_i blocks (BLOCK_Q rows)
//  - Separate K, V into K_j, V_j blocks (BLOCK_KV rows)
//  - For each Q_i, iterate through all K_j and V_j and accumulate the output.
//  - Use online softmax to merge result of each block that we don't need to save all score
//
// Result: Memory O(N) instead of O(N^2), HBM traffic O(N*d) instead of O(N^2)
// --------------------------------------------------------------------------

// Block sizes — adjust according to GPU SRAM size
// A100 has 192KB shared memory per SM
// BLOCK_Q * BLOCK_KV * sizeof(float) must be smaller than shared memory budget
#define BLOCK_Q 64
#define BLOCK_KV 64

// --------------------------------------------------------------------------
// Flash Attention kernel (single head, single batch element)
//
// Q, K, V: [N x d] — one head of one batch element
// O: [N x d] — output
// N: sequence length, d: head dimension
// --------------------------------------------------------------------------
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,   // [N x d]
    const float* __restrict__ K,   // [N x d]
    const float* __restrict__ V,   // [N x d]
    float*       __restrict__ O,   // [N x d]
    int N, int d,
    bool causal,
    float scale
) {
    // Each block handles BLOCK_Q rows of Q ("query block")
    int q_block_idx = blockIdx.x;
    int q_start = q_block_idx * BLOCK_Q;
    if (q_start >= N) return;
    int q_end = min(q_start + BLOCK_Q, N);

    // Shared memory layout:
    // sQ: [BLOCK_Q  x d] — current tile of Q
    // sK: [BLOCK_KV x d] — current tile of K
    // sV: [BLOCK_KV x d] — current tile of V
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + BLOCK_Q  * d;
    float* sV = sK + BLOCK_KV * d;

    // Running statistics per query row (In register, not using HBM)
    // m[i]: running max của attention scores cho query row i
    // l[i]: running sum của exp(scores - m) cho query row i
    float m[BLOCK_Q], l[BLOCK_Q];
    float acc[BLOCK_Q * 64];  // output accumulator — aasume that d <= 64

    // Initialization
    for (int i = 0; i < q_end - q_start; i++) {
        m[i] = -FLT_MAX;
        l[i] = 0.0f;
        for (int j = 0; j < d; j++) {
            acc[i * d + j] = 0.0f;
        }
    }

    // Load Q block into shared memory — load only once, use continuously
    for (int i = threadIdx.x; i < (q_end - q_start) * d; i += blockDim.x) {
        int qi = i / d;
        int di = i % d;
        sQ[qi * d + di] = Q[(q_start + qi) * d + di];
    }
    __syncthreads();

    // Iterate through each KV block
    
}