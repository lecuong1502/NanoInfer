#include "softmax.h"
#include <float.h>

// ------------------------------------------------------------------
// Warp-level reduction helpers (or often called warp-level primitives):
// are a set of intrinsic functions in CUDA designed to perform data aggregation operations
// between threads within the same warp (a group of 32 threads) extremely efficiently without using shared memory.
//
// __shfl_down_sync(): Retrieves data from the thread with the higher index. 
//This is the most important function for performing the Reduction operation (merging elements gradually).
//
// __shfl_xor_sync(): Exchanges data based on the thread index XOR operation, commonly used for butterfly networks.
//
// __shfl_up_sync() and __shfl_sync(): Other variations for moving data up or down a specific thread.

// __shfl_down_sync allows thread i to read register of thread "i + offset"
// Within the same warp — not via shared memory, latency ~4 cycles
// mask=0xffffffff means taht all 32 threads within warp attends
// -----------------------------------------------------------------

// Compute the sum of 32 threads in 1 warp, result is in thread 0
__device__ float warp_reduce_sum(float val) {
    // Each step reduces the number of active threads by half: 32→16→8→4→2→1
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Compute maximum all 32 threads in 1 warp, return at thread 0
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction: merge the result from multiple warps through shared memory
// Use when block has more than 1 warp (blockDim.x > 32)
__device__ float block_reduce_sum(float val) {
    // Maximize 32 warps per block (1024 threads / 32 = 32 warps)
    __shared__ float warp_sums[32];

    int lane = threadIdx.x % 32;    // position in warp (0-31)
    int warp_id = threadIdx.x / 32; // order of warp in block

    // Each warp reduces 1 units itself
    val = warp_reduce_sum(val);

    // First thread each warp writes the result to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    // First warp reads and reduces all warp_sums
    int num_warps = blockDim.x / 32;
    val = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;     // Only thread 0 of block has true resukt
}

__device__ float block_reduce_max(float val) {
    __shared__ float warp_maxs[32];

    int lane    = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0)
        warp_maxs[warp_id] = val;

    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = (threadIdx.x < num_warps) ? warp_maxs[lane] : -FLT_MAX;
    if (warp_id == 0)
        val = warp_reduce_max(val);

    return val;
}

// ------------------------------------------------------
// Naive Softmax — 3 passes through data
//
// Problem: exp(x) with large x makes overflow (FP32 max ~3.4e38)
// Example: exp(90) = 1.2e39 > FP32_MAX -> return inf
// This kernel is only used for benchmark, not for production
__global__ void softmax_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input  + row * cols;
    float* y = output + row * cols;

    // Pass 1: Compute sum of exp -> Easy to overflow with large x
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(x[i]);
    }
    sum = block_reduce_sum(sum);

    // Pass 2: Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = expf(x[i]) / sum;
    }
}

// -------------------------------------------------------
// Safe Softmax - 3 passes, numerically stable
//
// Trick: minus max before exp -> 0 <= exp(x_i - max) <= 1
// exp(x_i - max) never overflows because (x_i - max <= 0)
// Result is like naive but no inf/nan
// -------------------------------------------------------
__global__ void softmax_safe_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input  + row * cols;
    float* y = output + row * cols;

    // Pass 1: Find max to play anchor
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    max_val = block_reduce_max(max_val);

    // Broadcast max about threads through shared memory
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Pass 2: Compute sum(exp(x - max))
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    sum = block_reduce_sum(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // Pass 3: Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}

// -----------------------------------------------------------
// Online Softmax v2 — 1 pass, numerically stable, vectorized
//
// Optimizations vs v1:
//   1. float4 vectorized loads/stores  → 4x memory coalescing efficiency
//   2. #pragma unroll on warp/block reductions → zero loop overhead
//   3. inv_d = 1/d reciprocal → 1 rcp + N muls instead of N divisions
//   4. Adaptive block size (256 / 512) via template → better SM occupancy
// ----------------------------------------------------------
template <int BLOCK>
__global__ void softmax_online_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float*       y = output + row * cols;

    // ---- Pass 1: single-pass online (m, d) update with 2-way ILP ----
    // Two independent (m0,d0) and (m1,d1) streams process interleaved
    // elements so the GPU can pipeline expf calls without serial dependency.
    // float4 path: ONLY when cols % 4 == 0 (alignment guarantee).
    float m0 = -FLT_MAX, d0 = 0.0f;   // stream 0: even groups
    float m1 = -FLT_MAX, d1 = 0.0f;   // stream 1: odd  groups

    if ((cols & 3) == 0) {
        int cols4 = cols >> 2;
        // Stream 0: i = 0, 2*BLOCK, 4*BLOCK, ...
        // Stream 1: i = BLOCK, 3*BLOCK, 5*BLOCK, ...
        for (int i = threadIdx.x; i < cols4; i += 2 * BLOCK) {
            // Stream 0
            float4 v0 = reinterpret_cast<const float4*>(x)[i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float val = (&v0.x)[j];
                float mn  = fmaxf(m0, val);
                d0 = d0 * expf(m0 - mn) + expf(val - mn);
                m0 = mn;
            }
            // Stream 1 (independent — no dependency on stream 0)
            int i1 = i + BLOCK;
            if (i1 < cols4) {
                float4 v1 = reinterpret_cast<const float4*>(x)[i1];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float val = (&v1.x)[j];
                    float mn  = fmaxf(m1, val);
                    d1 = d1 * expf(m1 - mn) + expf(val - mn);
                    m1 = mn;
                }
            }
        }
    } else {
        // Scalar fallback for non-aligned cols (e.g. vocab size 50257)
        for (int i = threadIdx.x; i < cols; i += 2 * BLOCK) {
            float xi = x[i];
            float mn = fmaxf(m0, xi);
            d0 = d0 * expf(m0 - mn) + expf(xi - mn);
            m0 = mn;

            int i1 = i + BLOCK;
            if (i1 < cols) {
                float xi1 = x[i1];
                float mn1 = fmaxf(m1, xi1);
                d1 = d1 * expf(m1 - mn1) + expf(xi1 - mn1);
                m1 = mn1;
            }
        }
    }

    // Merge two streams into one (m, d)
    float m_new = fmaxf(m0, m1);
    float d = d0 * expf(m0 - m_new) + d1 * expf(m1 - m_new);
    float m = m_new;

    // ---- Step 1: warp-level reduce of (m, d) ----
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m_o = __shfl_down_sync(0xffffffff, m, offset);
        float d_o = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m_o);
        d = d * expf(m - m_new) + d_o * expf(m_o - m_new);
        m = m_new;
    }

    // ---- Step 2: block-level reduce via shared memory ----
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];

    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) { warp_m[warp_id] = m; warp_d[warp_id] = d; }
    __syncthreads();

    constexpr int NUM_WARPS = BLOCK / 32;
    if (warp_id == 0) {
        m = (lane < NUM_WARPS) ? warp_m[lane] : -FLT_MAX;
        d = (lane < NUM_WARPS) ? warp_d[lane] :  0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m_o = __shfl_down_sync(0xffffffff, m, offset);
            float d_o = __shfl_down_sync(0xffffffff, d, offset);
            float m_new = fmaxf(m, m_o);
            d = d * expf(m - m_new) + d_o * expf(m_o - m_new);
            m = m_new;
        }
        if (lane == 0) { warp_m[0] = m; warp_d[0] = d; }
    }
    __syncthreads();

    m = warp_m[0];
    d = warp_d[0];

    // ---- Pass 2: normalize with reciprocal (1 rcp + N muls) ----
    float inv_d = __frcp_rn(d);     // hardware reciprocal, ~1 cycle

    if ((cols & 3) == 0) {
        int cols4 = cols >> 2;
        for (int i = threadIdx.x; i < cols4; i += BLOCK) {
            float4 v = reinterpret_cast<const float4*>(x)[i];
            float4 o;
            o.x = expf(v.x - m) * inv_d;
            o.y = expf(v.y - m) * inv_d;
            o.z = expf(v.z - m) * inv_d;
            o.w = expf(v.w - m) * inv_d;
            reinterpret_cast<float4*>(y)[i] = o;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += BLOCK) {
            y[i] = expf(x[i] - m) * inv_d;
        }
    }
}

// ------------------------------------------------------------
// Host launch — adaptive block size based on row width
// ------------------------------------------------------------
void launch_softmax_online(
    const float* d_input,
    float* d_output,
    int rows, int cols
) {
    int blocks = rows;
    // 512 threads for wide rows (cols >= 1024) → better SM occupancy;
    // 256 threads for narrow rows → more blocks in flight per SM.
    if (cols >= 1024) {
        softmax_online_kernel<512><<<blocks, 512>>>(d_input, d_output, rows, cols);
    } else {
        softmax_online_kernel<256><<<blocks, 256>>>(d_input, d_output, rows, cols);
    }
}

void launch_softmax_safe(
    const float* d_input,
    float*       d_output,
    int rows, int cols
) {
    int threads = 256;
    int blocks  = rows;
    softmax_safe_kernel<<<blocks, threads>>>(d_input, d_output, rows, cols);
}

void launch_softmax_naive(
    const float* d_input,
    float*       d_output,
    int rows, int cols
) {
    int threads = 256;
    int blocks  = rows;
    softmax_naive_kernel<<<blocks, threads>>>(d_input, d_output, rows, cols);
}