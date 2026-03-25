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
    int num_kv_blocks = (N + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_KV;
        int kv_end = min(kv_start + BLOCK_KV, N);

        // Casual mask: Ignore all K/V blocks after the current Q block.
        // No need to compute attention with future tokens
        if (causal && kv_start > q_end - 1) break;

        // Load K block to shared memory
        for (int i = threadIdx.x; i < (kv_end - kv_start) * d; i += blockDim.x) {
            int ki = i / d;
            int di = i % d;
            sK[ki * d + di] = K[(kv_start + ki) * d + di];
        }

        // Load V block to shared memory
        for (int i = threadIdx.x; i < (kv_end - kv_start) * d; i += blockDim.x) {
            int vi = i / d;
            int di = i % d;
            sV[vi * d + di] = V[(kv_start + vi) * d + di];
        }

        __syncthreads();

        // Compute attention scores and update output (online softmax)
        // Each thread handles 1 query wor in Q block
        for (int qi = threadIdx.x; qi < q_end - q_start; qi += blockDim.x) {
            float m_new = m[qi];

            // Compute scores: S[qi, kj] = Q[qi] · K[kj] * scale
            // Find the new maximum simultaneously
            float scores[BLOCK_KV];
            for (int kj = 0; kj < kv_end - kv_start; kj++) {
                // Casual mask per-element: query in position (q_start + qi)
                // Only attend to key in position <= (q_start + qi)
                if (causal && (kv_start + kj) > (q_start + qi)) {
                    scores[kj] = -FLT_MAX;
                    continue;
                }

                float s = 0.0f;
                for (int di = 0; di < d; di++) {
                    s += sQ[qi * d + di] * sK[kj * d + di];
                }
                scores[kj] = s * scale;
                m_new = fmaxf(m_new, scores[kj]);
            }

            // Online softmax update:
            // Rescale ole accumulator according to new max
            float l_new = l[qi] * expf(m[qi] - m_new);
            for (int di = 0; di < d; di++) {
                acc[qi * d + di] *= expf(m[qi] - m_new);
            }

            // Add a contributrion from this KV Block
            for (int kj = 0; kj < kv_end - kv_start; kj++) {
                if (scores[kj] == -FLT_MAX) continue;   // masked out
                float p = expf(scores[kj] - m_new);
                l_new += p;
                for (int di = 0; di < d; di++) {
                    acc[qi * d + di] += p * sV[kj * d + di];
                }
            }

            // Update running statistics
            m[qi] = m_new;
            l[qi] = l_new;
        }
        __syncthreads();
    }

    // Write output to HBM — normalize by l (denominator of softmax)
    // This is the only part that writes to HBM in the entire kernel
    for (int qi = threadIdx.x; qi < q_end - q_start; qi += blockDim.x) {
        for (int di = 0; di < d; di++) {
            O[(q_start + qi) * d + di] = acc[qi * d + di] / l[qi];
        }
    }
}

// -------------------------------------------------------------
// Host launcher
// -------------------------------------------------------------
void launch_flash_attention(
    const float* d_Q,   // [B x H x N x d]
    const float* d_K,
    const float* d_V,
    float* d_O,
    int B, int H, int N, int d,
    bool causal
) {
    float scale = 1.0f / sqrtf((float)d);

    // Shared memory: Q tile + K tile + V tile
    size_t smem_bytes = (BLOCK_Q + 2 * BLOCK_KV) * d * sizeof(float);

    // Blocks: each block each Q-tile, multiple with B*H to cover the entire batch/heads
    int q_blocks = (N + BLOCK_Q - 1) / BLOCK_Q;
    int total_blocks = B * H * q_blocks;

    // 128 threads per block — each thread handles 1 query row in tile
    int threads = 128;

    // Kernel doesn't directly recieve B, H — use blockIdx to compute offset
    // Simplify: traverse B and H at the host (acceptable with small B and H)
    int stride_BH = N * d;  // stride among heads/batchs in flat array

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * stride_BH;
            flash_attention_kernel<<<q_blocks, threads, smem_bytes>>>(
                d_Q + offset,
                d_K + offset,
                d_V + offset,
                d_O + offset,
                N, d, causal, scale
            );
        }
    }
}