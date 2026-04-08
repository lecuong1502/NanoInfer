#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#include "../src/kernels/gemm.h"

// ---------------------------------------------------------------------------
// bench_gemm - compare NanoInfer tiled GEMM vs cuBLAS
//
// Standard measures:
//      1. Warmup 10 times for GPU to keep clock speed fine (avoid cold start bias)
//      2. Measure 100 times, takes average
//      3. Use cudaEvent (GPU timer) instead of clock() - microsecond
//
// Metric:
//      TFLOPS = (2 * M * N * K) / (elapsed_seconds * 1e12)
//      Each output element needs K multiply + K add = 2K operations
//
// Target (A100 80GB):
//      NanoInfer TILE=16: ~50-65% peak TFLOPS
//      NanoInfer TILE=32: ~70-80% peak TFLOPS
//      cuBLAS:            ~90-95% peak TFLOPS
// ---------------------------------------------------------------------------

#define WARMUP_ITERS  10
#define BENCH_ITERS   100

// ---------------------------------------------------------------------------
// Helper: allocate device matrix and fill with random floats [-1, 1]
// ---------------------------------------------------------------------------
static float* alloc_random(int rows, int cols) {
    int n = rows * cols;

    // Host buffer for random init
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// ---------------------------------------------------------------------------
// Helper: measure elapsed ms between two CUDA events over N iterations
// ---------------------------------------------------------------------------
static float measure_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ---------------------------------------------------------------------------
// Correctness check: max absolute difference between NanoInfer and cuBLAS
// If diff > threshold, something is wrong with the kernel
// ---------------------------------------------------------------------------
static float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// ---------------------------------------------------------------------------
// Run one benchmark configuration (M, N, K)
// ---------------------------------------------------------------------------
static void bench_one(cublasHandle_t cublas, int M, int N, int K) {
    float* d_A = alloc_random(M, K);
    float* d_B = alloc_random(K, N);
    float* d_C_nano   = nullptr;
    float* d_C_cublas = nullptr;
    cudaMalloc(&d_C_nano,   M * N * sizeof(float));
    cudaMalloc(&d_C_cublas, M * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double ops = 2.0 * M * N * K;  // total FLOPs for one GEMM

    // -----------------------------------------------------------------------
    // Correctness check (run once before benchmarking)
    // -----------------------------------------------------------------------
    cudaMemset(d_C_nano,   0, M * N * sizeof(float));
    cudaMemset(d_C_cublas, 0, M * N * sizeof(float));

    launch_gemm_tiled(d_A, d_B, d_C_nano, M, N, K);

    // cuBLAS: C = alpha * A @ B + beta * C
    // cuBLAS uses column-major by default, so we compute B^T @ A^T = (A@B)^T
    // which gives the correct row-major result in d_C_cublas
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,   // B [K x N] treated as column-major [N x K]
        d_A, K,   // A [M x K] treated as column-major [K x M]
        &beta,
        d_C_cublas, N);
    cudaDeviceSynchronize();

    float* h_nano   = (float*)malloc(M * N * sizeof(float));
    float* h_cublas = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(h_nano,   d_C_nano,   M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_nano, h_cublas, M * N);
    free(h_nano);
    free(h_cublas);

    // -----------------------------------------------------------------------
    // Benchmark NanoInfer GEMM
    // -----------------------------------------------------------------------

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        launch_gemm_tiled(d_A, d_B, d_C_nano, M, N, K);
    cudaDeviceSynchronize();

    // Measure
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        launch_gemm_tiled(d_A, d_B, d_C_nano, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float nano_ms   = measure_ms(start, stop) / BENCH_ITERS;
    double nano_tflops = ops / (nano_ms * 1e9);  // ms -> s -> TFLOPS

    // -----------------------------------------------------------------------
    // Benchmark cuBLAS
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; i++)
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_ms    = measure_ms(start, stop) / BENCH_ITERS;
    double cublas_tflops = ops / (cublas_ms * 1e9);

    double pct_peak = (nano_tflops / cublas_tflops) * 100.0;

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------
    printf("M=%-5d N=%-5d K=%-5d | "
           "NanoInfer %6.2f TFLOPS (%6.3f ms) | "
           "cuBLAS %6.2f TFLOPS (%6.3f ms) | "
           "%5.1f%% of cuBLAS | "
           "max_diff=%.2e %s\n",
           M, N, K,
           nano_tflops,   nano_ms,
           cublas_tflops, cublas_ms,
           pct_peak,
           diff,
           diff < 1e-3f ? "OK" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_nano);
    cudaFree(d_C_cublas);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ------------
// Main
// ------------
int main() {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f GB HBM)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (double)prop.totalGlobalMem / 1e9);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    printf("%-60s | %-40s | %-40s | %s | %s\n",
           "Size", "NanoInfer", "cuBLAS", "Ratio", "Diff");
    printf("%s\n", "--------------------------------------------------------------"
                   "--------------------------------------------------------------"
                   "--------------------");

    // Square matrices — common in transformer attention (seq x seq)
    printf("\n[Square matrices]\n");
    bench_one(cublas,  512,  512,  512);
    bench_one(cublas, 1024, 1024, 1024);
    bench_one(cublas, 2048, 2048, 2048);
    bench_one(cublas, 4096, 4096, 4096);

    // Transformer-realistic shapes — GPT-2 small (d=768, ffn=3072)
    printf("\n[Transformer shapes — GPT-2 (batch*seq=512)]\n");
    bench_one(cublas,  512,  768,  768);   // attention output proj
    bench_one(cublas,  512, 3072,  768);   // FFN up
    bench_one(cublas,  512,  768, 3072);   // FFN down
    bench_one(cublas,  512, 2304,  768);   // QKV fused (3*768)

    // Large — GPT-2 medium (d=1024) and large (d=1280)
    printf("\n[Larger models]\n");
    bench_one(cublas,  512, 1024, 1024);
    bench_one(cublas,  512, 4096, 1024);
    bench_one(cublas,  512, 1280, 1280);
    bench_one(cublas, 1024, 5120, 1280);

    cublasDestroy(cublas);
    return 0;
}