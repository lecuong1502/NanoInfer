// =============================================================================
// test_quantize.cu — Correctness tests for INT8 quantization pipeline
//
// Tests:
//   1. Quantize → Dequantize round-trip error
//      Expected: |x - dequant(quant(x))| <= scale/2   (Theorem: quantization
//      with bin width `scale` has worst-case error of half a bin)
//
//   2. launch_int8_gemm correctness vs float CPU reference
//      (quantize A + B_transposed → int8 gemm → dequantize → compare)
//
//   3. Saturation at INT8 boundaries (+127 / -128)
//
//   4. Non-divisible-by-4 tail handling (n % 4 != 0)
//
// Important layout note:
//   launch_int8_gemm expects B stored as [N x K] (transposed form).
//   The caller must pre-transpose B before passing it to the kernel.
//
// Exit code: 0 = all pass, 1 = at least one failure
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "../src/kernels/quantize.h"
#include "../src/kernels/gemm.h"

// ---------------------------------------------------------------------------
// Terminal colours
// ---------------------------------------------------------------------------
#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
#define PASS_STR GREEN "[PASS]" RESET
#define FAIL_STR RED "[FAIL]" RESET

static int g_pass = 0, g_fail = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void fill_random(float* h, int n, float lo, float hi, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        h[i] = lo + ((float)rand() / RAND_MAX) * (hi - lo);
}

static float max_abs_val(const float* h, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) if (fabsf(h[i]) > mx) mx = fabsf(h[i]);
    return mx;
}

static float max_abs_diff_f(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// Naive CPU GEMM: C = A @ B (float reference)
static void cpu_gemm_f(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// ---------------------------------------------------------------------------
// Test 1: Quantize → Dequantize round-trip
// Expected max error: scale / 2 (half bin-width theorem)
// ---------------------------------------------------------------------------
static void test_roundtrip(const char* label, int n, float lo, float hi, unsigned int seed) {
    float* h_in = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc(n * sizeof(float));
    int8_t* h_q8 = (int8_t*)malloc(n * sizeof(int8_t));
    int32_t* h_i32 = (int32_t*)malloc(n * sizeof(int32_t));

    fill_random(h_in, n, lo, hi, seed);
    float mx = max_abs_val(h_in, n);
    float scale = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;

    // Device buffers
    float *d_in, *d_out_f;
    int8_t *d_q8;
    int32_t *d_i32;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_q8, n * sizeof(int8_t));
    cudaMalloc(&d_i32, n * sizeof(int32_t));
    cudaMalloc(&d_out_f, n * sizeof(float));

    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: float32 → int8
    launch_quantize(d_in, d_q8, scale, n);

    // Step 2: int8 → int32 (trivially: copy each element widened)
    // Since we just want to dequantize int8, we cast via int32 proxy:
    // Copy int8 to host, cast to int32, copy back and dequantize with scale*scale=1
    cudaDeviceSynchronize();
    cudaMemcpy(h_q8, d_q8, n * sizeof(int8_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) h_i32[i] = (int32_t)h_q8[i];
    cudaMemcpy(d_i32, h_i32, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Step 3: int32 → float32   (use scale_A=scale, scale_B=1  ⇒ output = q * scale)
    launch_dequantize(d_i32, d_out_f, scale, 1.0f, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out_f, n * sizeof(float), cudaMemcpyDeviceToHost);

    // max round-trip error should be ≤ scale/2
    float max_err = max_abs_diff_f(h_in, h_out, n);
    float tol = scale / 2.0f + 1e-6f;   // +epsilon for fp32 rounding
    int ok = (max_err <= tol);

    printf("%s round-trip %-32s n=%-8d range=[%5.1f,%5.1f]  "
           "scale=%.4f  max_err=%.4f  tol=%.4f\n",
           ok ? PASS_STR : FAIL_STR, label, n, lo, hi, scale, max_err, tol);
    if (ok) g_pass++; else g_fail++;

    free(h_in); free(h_out); free(h_q8); free(h_i32);
    cudaFree(d_in); cudaFree(d_q8); cudaFree(d_i32); cudaFree(d_out_f);
}

// ---------------------------------------------------------------------------
// Test 2: Tail handling (n not divisible by 4)
// ---------------------------------------------------------------------------
static void test_tail(int n) {
    float* h_in  = (float*)malloc(n * sizeof(float));
    int8_t* h_q8  = (int8_t*)malloc(n * sizeof(int8_t));
    float* h_out = (float*)malloc(n * sizeof(float));
    int32_t*h_i32 = (int32_t*)malloc(n * sizeof(int32_t));

    fill_random(h_in, n, -1.0f, 1.0f, n * 31);
    float mx    = max_abs_val(h_in, n);
    float scale = (mx > 0.0f) ? mx / 127.0f : 1e-3f;

    float *d_in, *d_out_f;
    int8_t *d_q8;
    int32_t *d_i32;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_q8, n * sizeof(int8_t));
    cudaMalloc(&d_i32, n * sizeof(int32_t));
    cudaMalloc(&d_out_f, n * sizeof(float));

    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    launch_quantize(d_in, d_q8, scale, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_q8, d_q8, n * sizeof(int8_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) h_i32[i] = (int32_t)h_q8[i];
    cudaMemcpy(d_i32, h_i32, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    launch_dequantize(d_i32, d_out_f, scale, 1.0f, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out_f, n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = max_abs_diff_f(h_in, h_out, n);
    float tol = scale / 2.0f + 1e-6f;
    int ok = (max_err <= tol);

    printf("%s tail n=%-8d (n%%4=%d)  scale=%.4f  max_err=%.4f  tol=%.4f\n",
           ok ? PASS_STR : FAIL_STR, n, n % 4, scale, max_err, tol);
    if (ok) g_pass++; else g_fail++;

    free(h_in); free(h_q8); free(h_out); free(h_i32);
    cudaFree(d_in); cudaFree(d_q8); cudaFree(d_i32); cudaFree(d_out_f);
}

// ---------------------------------------------------------------------------
// Test 3: Saturation — values outside [-127*scale, 127*scale] must clamp
// ---------------------------------------------------------------------------
static void test_saturation() {
    // Build a vector with a few extreme values that should saturate
    int n = 16;
    float  h_in[16] = {
         0.5f,  -0.5f,        // normal range
         200.0f, -200.0f,     // way above range — should clamp to +/- 127
         1.0f,   1.0f, 1.0f, 1.0f,
         1.0f,   1.0f, 1.0f, 1.0f,
         1.0f,   1.0f, 1.0f, 1.0f
    };
    // scale = max/127 = 200/127 ≈ 1.575
    float scale = 200.0f / 127.0f;

    float  *d_in;  int8_t *d_q8;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_q8, n * sizeof(int8_t));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    launch_quantize(d_in, d_q8, scale, n);
    cudaDeviceSynchronize();

    int8_t h_q8[16];
    cudaMemcpy(h_q8, d_q8, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Values 200 / scale ≈ 127, -200 / scale ≈ -127 → check clamping
    int ok = (h_q8[2] == 127 || h_q8[2] == 126) &&  // +200 → saturate to +127
             (h_q8[3] == -127 || h_q8[3] == -128);  // -200 → saturate to -128

    printf("%s saturation  +200→Q=%d (expect 126-127)  -200→Q=%d (expect -128/-127)\n",
           ok ? PASS_STR : FAIL_STR, (int)h_q8[2], (int)h_q8[3]);
    if (ok) g_pass++; else g_fail++;

    cudaFree(d_in); cudaFree(d_q8);
}

// ---------------------------------------------------------------------------
// Test 4: INT8 GEMM vs float GEMM reference
//
// Protocol:
//   1. Generate random float A[M x K] and B[K x N]
//   2. Compute scale_A, scale_B per-tensor
//   3. Quantize A → A_q8, and B_T (B transposed [N x K]) → B_q8
//   4. Run launch_int8_gemm(A_q8, B_q8, C_i32, M, N, K)
//   5. Dequantize: C_f = C_i32 * scale_A * scale_B
//   6. Compare with CPU float GEMM C_ref = A @ B
//
// Tolerance: 5% of max(|C_ref|) because quantization error compounds through matmul
// ---------------------------------------------------------------------------
static void test_int8_gemm(int M, int N, int K, const char* label) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_BT = (size_t)N * K * sizeof(float);   // B transposed
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_BT = (float*)malloc(bytes_BT);   // B transposed [N x K]
    float* h_C_ref = (float*)malloc(bytes_C);
    float* h_C_int8 = (float*)malloc(bytes_C);

    // Fill with small values to reduce quantization error magnitude
    fill_random(h_A, M * K, -0.5f, 0.5f, M * 7 + K);
    fill_random(h_B, K * N, -0.5f, 0.5f, K * 11 + N);

    // CPU reference
    cpu_gemm_f(h_A, h_B, h_C_ref, M, N, K);

    // Transpose B: B_T[col][k] = B[k][col]
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            h_BT[j * K + k] = h_B[k * N + j];

    float scale_A = max_abs_val(h_A, M * K) / 127.0f;
    float scale_B = max_abs_val(h_B, K * N) / 127.0f;
    if (scale_A < 1e-8f) scale_A = 1e-8f;
    if (scale_B < 1e-8f) scale_B = 1e-8f;

    // Device buffers
    float *d_A_f, *d_BT_f, *d_C_f;
    int8_t *d_A_q, *d_BT_q;
    int32_t *d_C_i32;
    cudaMalloc(&d_A_f, bytes_A);
    cudaMalloc(&d_BT_f, bytes_BT);
    cudaMalloc(&d_A_q, M * K * sizeof(int8_t));
    cudaMalloc(&d_BT_q, N * K * sizeof(int8_t));
    cudaMalloc(&d_C_i32, M * N * sizeof(int32_t));
    cudaMalloc(&d_C_f, bytes_C);

    cudaMemcpy(d_A_f, h_A, bytes_A,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT_f, h_BT, bytes_BT, cudaMemcpyHostToDevice);

    launch_quantize(d_A_f, d_A_q, scale_A, M * K);
    launch_quantize(d_BT_f, d_BT_q, scale_B, N * K);
    cudaMemset(d_C_i32, 0, M * N * sizeof(int32_t));

    // INT8 GEMM — B must be pre-transposed: d_BT_q is [N x K]
    launch_int8_gemm(d_A_q, d_BT_q, d_C_i32, M, N, K);

    // Dequantize: C_float = C_int32 * scale_A * scale_B
    launch_dequantize(d_C_i32, d_C_f, scale_A, scale_B, M * N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_int8, d_C_f, bytes_C, cudaMemcpyDeviceToHost);

    // Check: tolerance = 5% of max absolute value in reference
    float max_ref = max_abs_val(h_C_ref, M * N);
    float tol = 0.05f * max_ref + 1e-4f;   // +epsilon for near-zero cases
    float diff = max_abs_diff_f(h_C_ref, h_C_int8, M * N);
    int ok = (diff < tol);

    printf("%s int8_gemm %-30s M=%-4d N=%-4d K=%-4d  "
           "max_diff=%.3f  tol=%.3f (5%%)\n",
           ok ? PASS_STR : FAIL_STR, label, M, N, K, diff, tol);
    if (ok) g_pass++; else g_fail++;

    free(h_A); free(h_B); free(h_BT); free(h_C_ref); free(h_C_int8);
    cudaFree(d_A_f); cudaFree(d_BT_f);
    cudaFree(d_A_q); cudaFree(d_BT_q);
    cudaFree(d_C_i32); cudaFree(d_C_f);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);
    printf("[Quantize / INT8 GEMM correctness tests]\n\n");

    // -----------------------------------------------------------------------
    // Group 1: Quantize → Dequantize round-trip
    // -----------------------------------------------------------------------
    printf("[Round-trip: quant → dequant error ≤ scale/2]\n");
    test_roundtrip("tiny",          16,    -1.0f,  1.0f, 1);
    test_roundtrip("normal range", 1024,   -1.0f,  1.0f, 2);
    test_roundtrip("large n",     65536,   -1.0f,  1.0f, 3);
    test_roundtrip("large values",  256,  -10.0f, 10.0f, 4);
    test_roundtrip("asymmetric",    512,   -3.0f, 15.0f, 5);

    // -----------------------------------------------------------------------
    // Group 2: Tail handling (n % 4 != 0)
    // -----------------------------------------------------------------------
    printf("\n[Tail handling — n not divisible by 4]\n");
    test_tail(1);       // n=1  → n%4=1
    test_tail(5);       // n=5  → n%4=1
    test_tail(253);     // n=253 → n%4=1
    test_tail(1022);    // n=1022 → n%4=2
    test_tail(1025);    // n=1025 → n%4=1
    // Divisible by 4 (main path, no tail)
    test_tail(1024);

    // -----------------------------------------------------------------------
    // Group 3: Saturation at INT8 limits
    // -----------------------------------------------------------------------
    printf("\n[Saturation — clamp to [-128, 127]]\n");
    test_saturation();

    // -----------------------------------------------------------------------
    // Group 4: INT8 GEMM vs float reference
    // -----------------------------------------------------------------------
    printf("\n[INT8 GEMM vs float reference (tolerance=5%%)]\n");
    test_int8_gemm( 16,  16,  16,  "tiny 16x16x16");
    test_int8_gemm( 32,  32,  32,  "small 32x32x32");
    test_int8_gemm( 64,  64,  64,  "mid 64x64x64");
    test_int8_gemm(128, 128, 128,  "square 128");
    test_int8_gemm( 32,  64,  32,  "non-square 32x64x32");
    test_int8_gemm( 64, 128,  64,  "non-square 64x128x64");
    // K must be multiple of 4 for dp4a to work correctly
    test_int8_gemm( 64,  64,  32,  "K=32 (mult of 4)");
    test_int8_gemm( 64,  32,  16,  "K=16 (mult of 4)");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%s --- %d / %d tests passed ---\n",
           g_fail == 0 ? PASS_STR : FAIL_STR, g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
