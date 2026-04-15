// =============================================================================
// test_softmax.cu — Correctness & numerical stability for all softmax variants
//
// Tests:
//   1. Row-sum correctness: each row must sum to 1.0 (± 1e-4)
//   2. Value range: all outputs in [0, 1], no NaN / Inf
//   3. Cross-variant agreement: online ≈ safe (max_diff < 1e-4)
//   4. Numerical stability: safe + online must survive large values (±90, ±500)
//      where naive overflows to Inf
//   5. Edge cases: single row, single element per row, vocab-size width
//
// Exit code: 0 = all pass, 1 = at least one failure
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "../src/kernels/softmax.h"

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
// Helpers
// ---------------------------------------------------------------------------
static void fill_random(float* h, int n, float lo, float hi, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        h[i] = lo + ((float)rand() / RAND_MAX) * (hi - lo);
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// Maximum absolute deviation of each row-sum from 1.0
static float max_row_sum_err(const float* out, int rows, int cols) {
    float mx = 0.0f;
    for (int r = 0; r < rows; r++) {
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += out[r * cols + c];
        float e = fabsf(s - 1.0f);
        if (e > mx) mx = e;
    }
    return mx;
}

// True iff all elements are finite and in [0, 1]
static int all_valid(const float* out, int n) {
    for (int i = 0; i < n; i++) {
        if (isnan(out[i]) || isinf(out[i])) return 0;
        if (out[i] < -1e-6f || out[i] > 1.0f + 1e-6f) return 0;
    }
    return 1;
}

// Run one softmax variant on the GPU and return a malloc'd host buffer
// variant: 0 = naive, 1 = safe, 2 = online
static float* gpu_softmax(int variant, const float* h_in, int rows, int cols) {
    size_t bytes = (size_t)rows * cols * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    switch (variant) {
        case 0: launch_softmax_naive (d_in, d_out, rows, cols); break;
        case 1: launch_softmax_safe  (d_in, d_out, rows, cols); break;
        case 2: launch_softmax_online(d_in, d_out, rows, cols); break;
    }
    cudaDeviceSynchronize();

    float* h_out = (float*)malloc(bytes);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return h_out;
}

// ---------------------------------------------------------------------------
// Test 1: Row-sum correctness + cross-variant agreement (safe vs online)
// ---------------------------------------------------------------------------
static void test_correctness(const char* label,
                             int rows, int cols,
                             float lo, float hi,
                             unsigned int seed) {
    int n = rows * cols;
    float* h_in    = (float*)malloc(n * sizeof(float));
    fill_random(h_in, n, lo, hi, seed);

    float* h_safe   = gpu_softmax(1, h_in, rows, cols);
    float* h_online = gpu_softmax(2, h_in, rows, cols);

    float safe_err   = max_row_sum_err(h_safe,   rows, cols);
    float online_err = max_row_sum_err(h_online, rows, cols);
    float cross_diff = max_abs_diff(h_safe, h_online, n);

    int ok_sum   = (safe_err < 1e-4f && online_err < 1e-4f);
    int ok_diff  = (cross_diff < 1e-4f);
    int ok_range = all_valid(h_safe, n) && all_valid(h_online, n);
    int ok = ok_sum && ok_diff && ok_range;

    printf("%s %-42s rows=%-5d cols=%-6d  "
           "sum_err(s/o)=%.1e/%.1e  cross=%.1e\n",
           ok ? PASS_STR : FAIL_STR, label, rows, cols,
           safe_err, online_err, cross_diff);
    if (!ok_range)
        printf("        " RED "[RANGE]" RESET " output has NaN/Inf or values outside [0,1]\n");
    if (ok) g_pass++; else g_fail++;

    free(h_in); free(h_safe); free(h_online);
}

// ---------------------------------------------------------------------------
// Test 2: Numerical stability (large inputs)
//   - naive may overflow → NaN/Inf is expected and NOT a failure for naive
//   - safe and online must remain valid regardless of input magnitude
// ---------------------------------------------------------------------------
static void test_stability(const char* label, int rows, int cols, float lo, float hi) {
    int n = rows * cols;
    float* h_in = (float*)malloc(n * sizeof(float));
    fill_random(h_in, n, lo, hi, 0xdeadbeef);

    float* h_naive = gpu_softmax(0, h_in, rows, cols);
    float* h_safe = gpu_softmax(1, h_in, rows, cols);
    float* h_online = gpu_softmax(2, h_in, rows, cols);

    // Naive overflow check (informational only — not a failure criterion)
    int naive_nan = 0;
    for (int i = 0; i < n; i++)
        if (isnan(h_naive[i]) || isinf(h_naive[i])) { naive_nan = 1; break; }

    int safe_ok   = all_valid(h_safe,   n);
    int online_ok = all_valid(h_online, n);
    float safe_err   = safe_ok   ? max_row_sum_err(h_safe,   rows, cols) : 999.f;
    float online_err = online_ok ? max_row_sum_err(h_online, rows, cols) : 999.f;

    int ok = safe_ok && online_ok && safe_err < 1e-4f && online_err < 1e-4f;

    printf("%s %-42s rows=%-3d cols=%-6d  range=[%g,%g]\n"
           "        naive_overflow=%-3s  safe=%s sum_err=%.1e  online=%s sum_err=%.1e\n",
           ok ? PASS_STR : FAIL_STR, label, rows, cols, lo, hi,
           naive_nan ? "YES" : "no",
           safe_ok   ? "OK" : RED "FAIL" RESET, safe_err,
           online_ok ? "OK" : RED "FAIL" RESET, online_err);
    if (ok) g_pass++; else g_fail++;

    free(h_in); free(h_naive); free(h_safe); free(h_online);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);
    printf("[Softmax correctness tests]\n\n");

    // -----------------------------------------------------------------------
    // Group 1: Correctness — safe and online must agree, rows sum to 1
    // -----------------------------------------------------------------------
    printf("[Correctness — row-sum ≈ 1.0 and safe ≈ online (tol=1e-4)]\n");
    test_correctness("single element",       1,      1, -1.0f,  1.0f, 1);
    test_correctness("single row col=32",    1,     32, -5.0f,  5.0f, 2);
    test_correctness("single row col=256",   1,    256, -2.0f,  2.0f, 3);
    test_correctness("attention 64x64",     64,     64, -1.0f,  1.0f, 4);
    test_correctness("attention 512x512",  512,    512, -1.0f,  1.0f, 5);
    test_correctness("attention 1024x1024",1024,  1024, -1.0f,  1.0f, 6);
    test_correctness("attention 2048x2048",2048,  2048, -1.0f,  1.0f, 7);
    test_correctness("vocab 1 x 50257",       1,  50257, -3.0f,  3.0f, 8);
    test_correctness("vocab 8 x 50257",       8,  50257, -3.0f,  3.0f, 9);
    test_correctness("vocab 32 x 50257",     32,  50257, -3.0f,  3.0f, 10);
    test_correctness("bandwidth 8192x8192", 8192,  8192, -1.0f,  1.0f, 11);

    // -----------------------------------------------------------------------
    // Group 2: Numerical stability
    //   - safe and online must not produce NaN/Inf with extreme inputs
    //   - naive is allowed to overflow (it is NOT numerically stable by design)
    // -----------------------------------------------------------------------
    printf("\n[Numerical stability — safe & online must survive large values]\n");
    test_stability("moderate  [-10, 10]",    16,   256,  -10.0f,  10.0f);
    test_stability("large     [-90, 90]",     8,   128,  -90.0f,  90.0f);
    test_stability("extreme   [-500, 500]",   4,    64, -500.0f, 500.0f);
    test_stability("very wide moderate",      2, 50257,   -5.0f,   5.0f);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%s --- %d / %d tests passed ---\n",
           g_fail == 0 ? PASS_STR : FAIL_STR, g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
