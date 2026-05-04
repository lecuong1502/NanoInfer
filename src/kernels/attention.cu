#include "attention.h"
#include <float.h>

// ---------------------------------------------------------------------------
// Flash Attention v1 — warp-per-row redesign
//
// Key insight: In the previous design, each THREAD computed the full dot
// product for one query row (64 serial iterations → GPU pipeline stalls).
// Now each WARP (32 threads) computes one dot product cooperatively:
//   - Each lane handles d/32 = 2 dimensions (for d=64)
//   - Warp-reduce via __shfl_down_sync → ~32× more parallelism in dot product
//
// Block layout:
//   BLOCK_Q  = 16 query rows per block
//   Threads  = BLOCK_Q * 32 = 512  (1 warp = 32 threads per query row)
//   SMEM     = (16 + 2×32) × 64 × 4 = 20 KB  (was 48 KB)
//   Concurrent blocks per SM: min(1536/512, 100KB/20KB) = min(3, 5) = 3
//   → 3 × 512 = 1536 threads per SM → 100% thread occupancy
//
// Grid: dim3(q_blocks, B*H) — one kernel launch covers all heads
// ---------------------------------------------------------------------------

#define BLOCK_Q  16   // query rows per block (= number of warps per block)
#define BLOCK_KV 32   // KV rows per tile

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int N, int d,
    bool causal,
    float scale
) {
    // ---- Decode head from grid ----
    int offset = blockIdx.y * N * d;
    const float* Qh = Q + offset;
    const float* Kh = K + offset;
    const float* Vh = V + offset;
    float*       Oh = O + offset;

    // ---- Q-tile bounds ----
    int q_start = blockIdx.x * BLOCK_Q;
    if (q_start >= N) return;
    int q_end = min(q_start + BLOCK_Q, N);
    int q_len = q_end - q_start;

    // ---- Warp/lane decomposition ----
    // warp_id = which query row this warp handles (0..BLOCK_Q-1)
    // lane    = thread's position within the warp (0..31)
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int qi      = warp_id;   // this warp's query row index in the tile

    // ---- Shared memory: sQ[BLOCK_Q x d] | sK[BLOCK_KV x d] | sV[BLOCK_KV x d] ----
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + BLOCK_Q  * d;
    float* sV = sK + BLOCK_KV * d;

    // ---- Per-warp accumulators (in registers — never touch HBM) ----
    // Each lane accumulates d/32 output dimensions.
    // For d=64: lanes 0..31 hold elements [lane, lane+32] of the output.
    // acc[k] = accumulator for dimension (lane + k*32), k=0,1,...
    const int ACC_SIZE = (64 + 31) / 32;  // = 2 for d<=64
    float acc[ACC_SIZE];
    for (int k = 0; k < ACC_SIZE; k++) acc[k] = 0.0f;

    float m_qi = -FLT_MAX;   // running max  (per warp, in registers)
    float l_qi = 0.0f;        // running denom

    // ---- Load Q tile cooperatively (all 512 threads work together) ----
    for (int i = threadIdx.x; i < q_len * d; i += blockDim.x) {
        sQ[(i / d) * d + (i % d)] = Qh[(q_start + i / d) * d + (i % d)];
    }
    __syncthreads();

    // ---- Main loop: iterate over KV tiles ----
    int num_kv_blocks = (N + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_KV;
        int kv_end   = min(kv_start + BLOCK_KV, N);
        int kv_len   = kv_end - kv_start;

        if (causal && kv_start > q_end - 1) break;

        // Load K tile cooperatively
        for (int i = threadIdx.x; i < kv_len * d; i += blockDim.x) {
            sK[(i / d) * d + (i % d)] = Kh[(kv_start + i / d) * d + (i % d)];
        }
        // Load V tile cooperatively
        for (int i = threadIdx.x; i < kv_len * d; i += blockDim.x) {
            sV[(i / d) * d + (i % d)] = Vh[(kv_start + i / d) * d + (i % d)];
        }
        __syncthreads();

        // Only warps with valid query rows do work
        if (qi < q_len) {
            float m_new = m_qi;
            float scores[BLOCK_KV];

            // ---- Compute QK^T scores — warp-parallel dot product ----
            for (int kj = 0; kj < kv_len; kj++) {
                if (causal && (kv_start + kj) > (q_start + qi)) {
                    scores[kj] = -FLT_MAX;
                    continue;
                }
                // Each lane accumulates d/32 multiply-adds
                float s = 0.0f;
                for (int di = lane; di < d; di += 32) {
                    s += sQ[qi * d + di] * sK[kj * d + di];
                }
                // Warp reduce: lane 0 gets the full dot product
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    s += __shfl_down_sync(0xffffffff, s, off);
                }
                // Broadcast score to all lanes in warp
                scores[kj] = __shfl_sync(0xffffffff, s, 0) * scale;
                m_new = fmaxf(m_new, scores[kj]);
            }

            // ---- Rescale old accumulator with new max (precomputed once) ----
            float corr = expf(m_qi - m_new);
            l_qi *= corr;
            for (int k = 0; k < ACC_SIZE; k++) acc[k] *= corr;

            // ---- Add this KV block's contribution ----
            for (int kj = 0; kj < kv_len; kj++) {
                if (scores[kj] == -FLT_MAX) continue;
                float p = expf(scores[kj] - m_new);
                l_qi += p;
                // Each lane accumulates its d/32 dimensions
                for (int di = lane, k = 0; di < d; di += 32, k++) {
                    acc[k] += p * sV[kj * d + di];
                }
            }

            m_qi = m_new;
        }
        __syncthreads();
    }

    // ---- Write output: each lane writes its d/32 elements ----
    if (qi < q_len) {
        float inv_l = __frcp_rn(l_qi);
        for (int di = lane, k = 0; di < d; di += 32, k++) {
            Oh[(q_start + qi) * d + di] = acc[k] * inv_l;
        }
    }
}

// -------------------------------------------------------------
// Host launcher
// -------------------------------------------------------------
void launch_flash_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int B, int H, int N, int d,
    bool causal
) {
    float scale = 1.0f / sqrtf((float)d);

    // SMEM: sQ + sK + sV
    size_t smem_bytes = (size_t)(BLOCK_Q + 2 * BLOCK_KV) * d * sizeof(float);

    // Allow up to 100 KB shared memory per block (sm_89 supports 102400 bytes)
    cudaFuncSetAttribute(flash_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    // 1 warp per query row → BLOCK_Q warps per block
    int threads = BLOCK_Q * 32;   // 512

    // Grid: x = Q-tiles, y = B*H (all heads in ONE kernel launch)
    int q_blocks = (N + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(q_blocks, B * H);

    flash_attention_kernel<<<grid, threads, smem_bytes>>>(
        d_Q, d_K, d_V, d_O,
        N, d, causal, scale
    );
}