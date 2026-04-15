// =============================================================================
// test_gemm.cu — Correctness tests for launch_gemm_tiled
//
// Protocol:
//   1. Generate random A[M x K], B[K x N] on CPU
//   2. Compute C_ref = A @ B on CPU (naive O(MNK))
//   3. Run launch_gemm_tiled on GPU, copy result back
//   4. Check max_abs_diff(C_ref, C_gpu) < TOL
//
// Tests cover:
//   - TILE_16 path  (M, N, K < 512)
//   - TILE_32 path  (M, N, K >= 512)
//   - Non-square matrices
//   - K not multiple of TILE_SIZE (boundary/padding correctness)
//   - Transformer-realistic shapes (GPT-2)
//
// Exit code: 0 = all pass, 1 = at least one failure
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../src/kernels/gemm.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define TOL 1e-3f

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
#define PASS_STR GREEN "[PASS]" RESET
#define FAIL_STR RED "[FAIL]" RESET

static int g_pass = 0, g_fail = 0;

static void fill_random(float* h, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // uniform [-1, 1]
}

// Naive CPU GEMM: C = A @ B
static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ---------------------------------------------------------------------------
// Single test case
// ---------------------------------------------------------------------------
static void test_case(int M, int N, int K, const char* label) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C_ref = (float*)malloc(bytes_C);
    float* h_C_gpu = (float*)malloc(bytes_C);

    fill_random(h_A, M * K, (unsigned)(M * 7 + K * 3 + N));
    fill_random(h_B, K * N, (unsigned)(K * 5 + N * 11 + M));

    // CPU reference
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    // GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes_C);

    launch_gemm_tiled(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(FAIL_STR " %-45s M=%-5d N=%-5d K=%-5d  CUDA_ERROR=%s\n",
               label, M, N, K, cudaGetErrorString(err));
        g_fail++;
        goto cleanup;
    }

    cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost);

    {
        float diff = max_abs_diff(h_C_ref, h_C_gpu, M * N);
        bool  ok   = (diff < TOL);
        printf("%s %-45s M=%-5d N=%-5d K=%-5d  max_diff=%.2e\n",
               ok ? PASS_STR : FAIL_STR, label, M, N, K, diff);
        if (ok) g_pass++; else g_fail++;
    }

cleanup:
    free(h_A); free(h_B); free(h_C_ref); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);
    printf("[GEMM correctness tests — TOL=%.0e]\n\n", (double)TOL);

    // -----------------------------------------------------------------------
    // Small matrices — exercises the TILE_16 path (M,N,K < 512)
    // -----------------------------------------------------------------------
    printf("[Small / TILE_16 path]\n");
    test_case(16, 16, 16, "tiny square (16x16x16)");
    test_case(32, 32, 32, "square 32");
    test_case(64, 64, 64, "square 64");
    test_case(128, 128, 128, "square 128");
    test_case(256, 256, 256, "square 256");

    // Non-square — common in embedding / projection layers
    test_case(32, 64, 16, "non-square 32x64x16");
    test_case(64, 128, 32, "non-square 64x128x32");
    test_case(128, 64, 256, "non-square 128x64x256");

    // K not a multiple of TILE_SIZE — tests zero-padding boundary handling
    test_case(32, 32, 17, "K=17 (non-tile-aligned)");
    test_case(64, 64, 37, "K=37 (non-tile-aligned)");
    test_case(128, 128, 97, "K=97 (non-tile-aligned)");

    // -----------------------------------------------------------------------
    // Large matrices — exercises the TILE_32 path (M,N,K >= 512)
    // -----------------------------------------------------------------------
    printf("\n[Large / TILE_32 path]\n");
    test_case(512, 512, 512, "square 512");
    test_case(1024, 1024, 1024, "square 1024");

    // GPT-2 small realistic shapes (batch*seq = 512, d_model = 768, d_ffn = 3072)
    printf("\n[GPT-2 transformer shapes (batch*seq=512)]\n");
    test_case(512, 768, 768, "QKV output projection");
    test_case(512, 3072, 768, "FFN up-projection");
    test_case(512, 768, 3072, "FFN down-projection");
    test_case(512, 2304, 768, "QKV fused (3*d_model)");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%s --- %d / %d tests passed ---\n",
           g_fail == 0 ? PASS_STR : FAIL_STR, g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
