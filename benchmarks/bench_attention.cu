#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/kernels/attention.h"

// ---------------------------------------------------------------------------
// bench_attention — Flash Attention v1 vs Naive Attention
//
// Naive attention:
//   S = Q @ K^T          → [B, H, N, N]  materializes full score matrix in HBM
//   P = softmax(S/sqrt(d))
//   O = P @ V            → [B, H, N, d]
//   Memory: O(N²) — at N=4096, H=12, B=8 → ~12 GB just for score matrix
//
// Flash Attention:
//   Tiles Q/K/V into SRAM blocks, computes attention without writing N×N to HBM
//   Memory: O(N) — score matrix never materializes in HBM
//
// Key metrics:
//   Latency (ms)     — wall-clock time per call
//   Memory (MB)      — peak HBM allocation for intermediate tensors
//   HBM reads (GB/s) — bandwidth efficiency
// ---------------------------------------------------------------------------

#define WARMUP_ITERS 10
#define BENCH_ITERS  50   // fewer iters — attention is slower than GEMM

static float* alloc_random(int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX) * 0.1f;   // small values to avoid softmax overflow
    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// ---------------------------------------------------------------------------
// Naive attention implementation (reference, CPU-friendly, not optimized)
// Used only for correctness comparison on small inputs
// ---------------------------------------------------------------------------
static void naive_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int S, int d
) {
    float scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
        int offset = (b * H + h) * S * d;
        const float* q = Q + offset;
        const float* k = K + offset;
        const float* v = V + offset;
        float*       o = O + offset;

        // Score matrix [S x S]
        float* scores = (float*)malloc(S * S * sizeof(float));

        // S = Q @ K^T * scale
        for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            // Causal mask: only attend to j <= i
            if (j > i) { scores[i * S + j] = -1e9f; continue; }
            float dot = 0.0f;
            for (int kk = 0; kk < d; kk++)
                dot += q[i * d + kk] * k[j * d + kk];
            scores[i * S + j] = dot * scale;
        }

        // Softmax per row
        for (int i = 0; i < S; i++) {
            float mx = -1e9f;
            for (int j = 0; j <= i; j++)
                if (scores[i * S + j] > mx) mx = scores[i * S + j];
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * S + j] = expf(scores[i * S + j] - mx);
                sum += scores[i * S + j];
            }
            for (int j = 0; j <= i; j++)
                scores[i * S + j] /= sum;
        }

        // O = P @ V
        for (int i = 0; i < S; i++)
        for (int kk = 0; kk < d; kk++) {
            float acc = 0.0f;
            for (int j = 0; j <= i; j++)
                acc += scores[i * S + j] * v[j * d + kk];
            o[i * d + kk] = acc;
        }

        free(scores);
    }
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
// Benchmark one configuration
// ---------------------------------------------------------------------------
static void bench_one(int B, int H, int S, int d) {
    int qkv_n = B * H * S * d;
    size_t qkv_bytes = (size_t)qkv_n * sizeof(float);

    float* d_Q = alloc_random(qkv_n);
    float* d_K = alloc_random(qkv_n);
    float* d_V = alloc_random(qkv_n);
    float* d_O_flash = nullptr;
    cudaMalloc(&d_O_flash, qkv_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------------------------------------------------
    // Correctness check (small S only — naive is O(N²) on CPU)
    // -----------------------------------------------------------------------
    const char* correct_str = "skip";
    float diff = 0.0f;
    if (S <= 128 && B == 1) {
        float* h_Q = (float*)malloc(qkv_bytes);
        float* h_K = (float*)malloc(qkv_bytes);
        float* h_V = (float*)malloc(qkv_bytes);
        float* h_O_ref   = (float*)calloc(qkv_n, sizeof(float));
        float* h_O_flash = (float*)malloc(qkv_bytes);

        cudaMemcpy(h_Q, d_Q, qkv_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K, d_K, qkv_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V, d_V, qkv_bytes, cudaMemcpyDeviceToHost);

        naive_attention_cpu(h_Q, h_K, h_V, h_O_ref, B, H, S, d);

        launch_flash_attention(d_Q, d_K, d_V, d_O_flash, B, H, S, d, true);
        cudaDeviceSynchronize();
        cudaMemcpy(h_O_flash, d_O_flash, qkv_bytes, cudaMemcpyDeviceToHost);

        diff = max_abs_diff(h_O_ref, h_O_flash, qkv_n);
        correct_str = diff < 1e-3f ? "OK" : "FAIL";

        free(h_Q); free(h_K); free(h_V);
        free(h_O_ref); free(h_O_flash);
    }

    // -----------------------------------------------------------------------
    // Benchmark Flash Attention
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; i++)
        launch_flash_attention(d_Q, d_K, d_V, d_O_flash, B, H, S, d, true);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++)
        launch_flash_attention(d_Q, d_K, d_V, d_O_flash, B, H, S, d, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float flash_ms;
    cudaEventElapsedTime(&flash_ms, start, stop);
    flash_ms /= BENCH_ITERS;

    // FLOPs: 2 * B * H * S * S * d (Q@K^T) + 2 * B * H * S * S * d (P@V)
    double flops = 4.0 * B * H * S * S * d;
    double flash_tflops = flops / (flash_ms * 1e9);

    // Memory: Flash reads Q,K,V once = 3 * qkv_bytes, writes O = qkv_bytes
    double flash_bw_gb = (4.0 * qkv_bytes) / (flash_ms * 1e6);

    // Naive memory would be: Q,K,V reads + N×N score matrix writes/reads
    double naive_score_mb = (double)B * H * S * S * 4 / 1e6;

    printf("B=%-2d H=%-3d S=%-5d d=%-3d | "
           "flash %6.2f ms %5.2f TFLOPS %6.1f GB/s | "
           "score_matrix_if_naive=%.0f MB | "
           "correct=%s\n",
           B, H, S, d,
           flash_ms, flash_tflops, flash_bw_gb,
           naive_score_mb,
           correct_str);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O_flash);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    printf("[Flash Attention benchmark]\n");
    printf("d_head=64 (GPT-2 style)\n\n");

    // GPT-2 small: H=12, d=64
    printf("[GPT-2 small — H=12, d=64]\n");
    bench_one(1,  12,  128, 64);
    bench_one(1,  12,  512, 64);
    bench_one(1,  12, 1024, 64);
    bench_one(8,  12,  512, 64);
    bench_one(8,  12, 1024, 64);
    bench_one(8,  12, 2048, 64);

    // Long context — where Flash Attention advantage is most visible
    printf("\n[Long context — memory advantage]\n");
    bench_one(1,  12, 4096, 64);
    bench_one(1,  12, 8192, 64);
    bench_one(4,  12, 4096, 64);

    // LLaMA-style: H=32, d=128
    printf("\n[LLaMA style — H=32, d=128]\n");
    bench_one(1,  32,  512, 128);
    bench_one(1,  32, 2048, 128);
    bench_one(4,  32, 2048, 128);

    return 0;
}