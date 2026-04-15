// =============================================================================
// test_attention.cu — Correctness tests for Flash Attention v1
//
// Strategy: compare GPU Flash Attention against a naive CPU implementation
// (same algorithm as bench_attention.cu reference). Tolerance is loose (2e-2)
// because Flash Attention accumulates online updates that introduce small
// floating-point reordering vs the CPU's sequential row-by-row computation.
//
// NOTE: The Flash Attention kernel in attention.cu has a fixed register array:
//   float acc[BLOCK_Q * 64]   (line in .cu: assumes d_head <= 64)
// All tests use d=64 to stay within this limit.
//
// Tests:
//   - Causal (autoregressive) attention
//   - Non-causal (bidirectional) attention
//   - Single-head and multi-head
//   - Multi-batch
//   - Various sequence lengths
//
// Exit code: 0 = all pass, 1 = at least one failure
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/kernels/attention.h"

// ---------------------------------------------------------------------------
// Terminal colours
// ---------------------------------------------------------------------------
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"
#define PASS_STR GREEN "[PASS]" RESET
#define FAIL_STR RED   "[FAIL]" RESET

static int g_pass = 0, g_fail = 0;

// ---------------------------------------------------------------------------
// CPU naive attention — identical reference used in bench_attention.cu
// Q, K, V, O: [B x H x S x d] stored as flat row-major array
// ---------------------------------------------------------------------------
static void naive_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int S, int d, int causal
) {
    float scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
        int offset           = (b * H + h) * S * d;
        const float* q       = Q + offset;
        const float* k       = K + offset;
        const float* v       = V + offset;
        float*       o       = O + offset;

        float* scores = (float*)malloc((size_t)S * S * sizeof(float));

        // S = Q @ K^T * scale
        for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            if (causal && j > i) { scores[i * S + j] = -1e9f; continue; }
            float dot = 0.0f;
            for (int kk = 0; kk < d; kk++)
                dot += q[i * d + kk] * k[j * d + kk];
            scores[i * S + j] = dot * scale;
        }

        // Row-wise softmax
        for (int i = 0; i < S; i++) {
            float mx = -1e9f;
            for (int j = 0; j < S; j++)
                if (scores[i * S + j] > mx) mx = scores[i * S + j];
            float sum = 0.0f;
            for (int j = 0; j < S; j++) {
                scores[i * S + j] = expf(scores[i * S + j] - mx);
                sum += scores[i * S + j];
            }
            for (int j = 0; j < S; j++) scores[i * S + j] /= sum;
        }

        // O = P @ V
        for (int i = 0; i < S; i++)
        for (int kk = 0; kk < d; kk++) {
            float acc = 0.0f;
            for (int j = 0; j < S; j++)
                acc += scores[i * S + j] * v[j * d + kk];
            o[i * d + kk] = acc;
        }

        free(scores);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void fill_random(float* h, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        // Small values in [-0.1, 0.1] — avoids attention score overflow in softmax
        h[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

// ---------------------------------------------------------------------------
// Single test case
// ---------------------------------------------------------------------------
static void test_case(int B, int H, int S, int d, int causal, float tol, const char* label) {
    int    n     = B * H * S * d;
    size_t bytes = (size_t)n * sizeof(float);

    float* h_Q    = (float*)malloc(bytes);
    float* h_K    = (float*)malloc(bytes);
    float* h_V    = (float*)malloc(bytes);
    float* h_O_ref = (float*)calloc(n, sizeof(float));
    float* h_O_gpu = (float*)malloc(bytes);

    fill_random(h_Q, n, (unsigned)(B * 100 + H * 10 + S));
    fill_random(h_K, n, (unsigned)(B * 200 + H * 20 + S));
    fill_random(h_V, n, (unsigned)(B * 300 + H * 30 + S));

    // CPU reference
    naive_attention_cpu(h_Q, h_K, h_V, h_O_ref, B, H, S, d, causal);

    // GPU flash attention
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes); cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes); cudaMalloc(&d_O, bytes);

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, bytes);

    launch_flash_attention(d_Q, d_K, d_V, d_O, B, H, S, d, (bool)causal);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(FAIL_STR " %-45s CUDA_ERROR=%s\n",
               label, cudaGetErrorString(err));
        g_fail++;
        goto cleanup;
    }

    cudaMemcpy(h_O_gpu, d_O, bytes, cudaMemcpyDeviceToHost);

    {
        float diff = max_abs_diff(h_O_ref, h_O_gpu, n);
        int   ok   = (diff < tol);
        printf("%s %-45s B=%-2d H=%-3d S=%-5d d=%d causal=%c  diff=%.2e (tol=%.0e)\n",
               ok ? PASS_STR : FAIL_STR, label,
               B, H, S, d, causal ? 'Y' : 'N', diff, (double)tol);
        if (ok) g_pass++; else g_fail++;
    }

cleanup:
    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_gpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);
    printf("[Flash Attention correctness tests]\n");
    printf("(tolerance=2e-2 — online softmax accumulation vs CPU sequential)\n\n");

    // NOTE: All tests use d=64.  The kernel has a compile-time array
    //       float acc[BLOCK_Q * 64] which limits d_head to 64.

    // -----------------------------------------------------------------------
    // Causal attention (autoregressive generation)
    // -----------------------------------------------------------------------
    printf("[Causal attention]\n");
    test_case(1,  1,  16, 64, 1, 2e-2f, "1head S=16");
    test_case(1,  1,  32, 64, 1, 2e-2f, "1head S=32");
    test_case(1,  1,  64, 64, 1, 2e-2f, "1head S=64");
    test_case(1,  1, 128, 64, 1, 2e-2f, "1head S=128");
    test_case(1, 12,  64, 64, 1, 2e-2f, "GPT-2 H=12 S=64");
    test_case(1, 12, 128, 64, 1, 2e-2f, "GPT-2 H=12 S=128");
    test_case(1, 12, 256, 64, 1, 3e-2f, "GPT-2 H=12 S=256");

    // -----------------------------------------------------------------------
    // Non-causal (bidirectional, e.g. BERT)
    // -----------------------------------------------------------------------
    printf("\n[Non-causal attention]\n");
    test_case(1,  1,  32, 64, 0, 2e-2f, "non-causal S=32");
    test_case(1,  1,  64, 64, 0, 2e-2f, "non-causal S=64");
    test_case(1,  1, 128, 64, 0, 2e-2f, "non-causal S=128");
    test_case(1,  4,  64, 64, 0, 2e-2f, "non-causal H=4 S=64");

    // -----------------------------------------------------------------------
    // Multi-batch
    // -----------------------------------------------------------------------
    printf("\n[Multi-batch]\n");
    test_case(2,  1,  32, 64, 1, 2e-2f, "B=2,H=1");
    test_case(4,  1,  64, 64, 1, 2e-2f, "B=4,H=1");
    test_case(2, 12,  64, 64, 1, 2e-2f, "B=2,GPT-2 H=12");
    test_case(4, 12,  64, 64, 1, 2e-2f, "B=4,GPT-2 H=12");

    // -----------------------------------------------------------------------
    // Edge: sequence length = 1 (decode step, auto-regressive)
    // -----------------------------------------------------------------------
    printf("\n[Edge — single token decode step]\n");
    test_case(1,  1,   1, 64, 1, 1e-5f, "S=1 causal");
    test_case(1, 12,   1, 64, 1, 1e-5f, "S=1 H=12");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%s --- %d / %d tests passed ---\n",
           g_fail == 0 ? PASS_STR : FAIL_STR, g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
