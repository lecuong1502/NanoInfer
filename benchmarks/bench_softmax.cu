#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "../src/kernels/softmax.h"

// ---------------------------------------------------------------------------
// bench_softmax - Compare 3 variants of softmax:
//   1. Naive   (3 pass, no stable)
//   2. Safe    (3 pass, subtract max before)
//   3. Online  (1 pass, numerically stable) ← NanoInfer default
//
// Key metric: GB/s (memory bandwidth) is more important than FLOPS because softmax
// is memory-bound operation - bottleneck is read/write HBM, not computing
//
// Theory:
//   Naive  reads:  3 * rows * cols * 4 bytes (3 passes)
//   Safe   reads:  3 * rows * cols * 4 bytes (3 passes)
//   Online reads:  2 * rows * cols * 4 bytes (1 pass read + 1 pass normalize)
//  → Online theory is 1.5x faster in bandwidth
// ---------------------------------------------------------------------------

#define WARMUP_ITERS 10
#define BENCH_ITERS  100

static float* alloc_random(int rows, int cols) {
    int n = rows * cols;
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;  // range [-5, 5]
    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// Verify correctness: online output should match safe output within tolerance
static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void bench_one(int rows, int cols) {
    int n = rows * cols;
    size_t bytes = (size_t)n * sizeof(float);

    float* d_in = alloc_random(rows, cols);
    float* d_out_naive = nullptr;
    float* d_out_safe = nullptr;
    float* d_out_online = nullptr;
    cudaMalloc(&d_out_naive, bytes);
    cudaMalloc(&d_out_safe, bytes);
    cudaMalloc(&d_out_online, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------------------------------------------------
    // Correctness check
    // -----------------------------------------------------------------------
    launch_softmax_naive (d_in, d_out_naive, rows, cols);
    launch_softmax_safe (d_in, d_out_safe, rows, cols);
    launch_softmax_online(d_in, d_out_online, rows, cols);
    cudaDeviceSynchronize();

    float* h_safe   = (float*)malloc(bytes);
    float* h_online = (float*)malloc(bytes);
    cudaMemcpy(h_safe,   d_out_safe,   bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_online, d_out_online, bytes, cudaMemcpyDeviceToHost);
    float diff = max_abs_diff(h_safe, h_online, n);
    free(h_safe);
    free(h_online);

    // -----------------------------------------------------------------------
    // Benchmark naive
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; i++)
        launch_softmax_naive(d_in, d_out_naive, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        launch_softmax_naive(d_in, d_out_naive, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms;
    cudaEventElapsedTime(&naive_ms, start, stop);
    naive_ms /= BENCH_ITERS;
    // Naive: 3 full passes → 3 reads + 1 write = 4N floats
    double naive_bw = (4.0 * bytes) / (naive_ms * 1e6);  // GB/s

    // -----------------------------------------------------------------------
    // Benchmark safe
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; i++)
        launch_softmax_safe(d_in, d_out_safe, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        launch_softmax_safe(d_in, d_out_safe, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float safe_ms;
    cudaEventElapsedTime(&safe_ms, start, stop);
    safe_ms /= BENCH_ITERS;
    double safe_bw = (4.0 * bytes) / (safe_ms * 1e6);

    // -----------------------------------------------------------------------
    // Benchmark online
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; i++)
        launch_softmax_online(d_in, d_out_online, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        launch_softmax_online(d_in, d_out_online, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float online_ms;
    cudaEventElapsedTime(&online_ms, start, stop);
    online_ms /= BENCH_ITERS;
    // Online: 1 read pass + 1 normalize pass = 2N floats
    double online_bw = (2.0 * bytes) / (online_ms * 1e6);

    double speedup = naive_ms / online_ms;

    printf("rows=%-6d cols=%-6d | "
           "naive %6.3f ms (%5.1f GB/s) | "
           "safe %6.3f ms (%5.1f GB/s) | "
           "online %6.3f ms (%5.1f GB/s) | "
           "speedup=%.2fx | diff=%.2e %s\n",
           rows, cols,
           naive_ms,  naive_bw,
           safe_ms,   safe_bw,
           online_ms, online_bw,
           speedup,
           diff,
           diff < 1e-5f ? "OK" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out_naive);
    cudaFree(d_out_safe);
    cudaFree(d_out_online);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | Peak HBM bandwidth: %.0f GB/s\n\n",
           prop.name,
           // Approximate peak bandwidth from memory clock and bus width
           (double)prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9);

    printf("[Softmax benchmark — rows x cols]\n\n");

    // Small: attention softmax (seq x seq)
    printf("[Attention softmax — seq x seq]\n");
    bench_one(512,    512);
    bench_one(1024,  1024);
    bench_one(2048,  2048);
    bench_one(4096,  4096);

    // Wide rows: vocab-size softmax at LM head
    printf("\n[LM head softmax — batch x vocab]\n");
    bench_one(1,    50257);   // single token generation
    bench_one(8,    50257);   // batch=8
    bench_one(32,   50257);   // batch=32

    // Memory bandwidth stress test
    printf("\n[Bandwidth stress]\n");
    bench_one(8192,  8192);
    bench_one(16384, 4096);

    return 0;
}