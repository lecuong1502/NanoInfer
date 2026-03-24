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
// Online Softmax — 1 pass, numerically stable
// 
// Key Insight: m_new = max(m_old, x_i)
//              d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)

// Prove correctness: end the loop, return denominator of Safe Softmax
// d = sum_i exp(x_i - m) with m = max(x_0, ..., x_{N-1})
// This is the Flash Attention platform
// ----------------------------------------------------------
__global__ void softmax_online_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input  + row * cols;
    float* y = output + row * cols;

    // Each thread maintains running private (max, denominator)
    float m = -FLT_MAX;     // running max
    float d = 0.0f;         // running denominator = sum(exp(x_i - m))

    // Only Pass: Update (m, d) for each element
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        // Rescale old d according to max before add exp(xi - n_new)
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // Merge (m, d) from all threads - Need to process both max and sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m_other = __shfl_down_sync(0xffffffff, m, offset);
        float d_other = __shfl_down_sync(0xffffffff, d, offset);
        // Merge 2 running stats: select larger max, rescale another
        float m_new = fmaxf(m, m_other);
        d = d * expf(m - m_new) + d_other * expf(m_other - m_new);
        m = m_new;
    }

    // Broadcast result from thread 0 to all threads in warp
    m = __shfl_sync(0xffffffff, m, 0);
    d = __shfl_sync(0xffffffff, d, 0);

    // Normalize: Reread x once to compute output
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - m) / d;
    }
}

// ------------------------------------------------------------
// Host launch — use online softmax (single-pass) as default
// ------------------------------------------------------------
void launch_softmax_online (
    const float* d_input,
    float* d_output,
    int rows, int cols
) {
    // Each block processes 1 row — 256 threads per block (8 warps)
    // cols > 256: each thread processes multiple elements through stride loop
    int threads = 256;
    int blocks = rows;
    softmax_online_kernel<<<blocks, threads>>>(d_input, d_output, rows, cols);
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