#include "quantize.h"
#include <float.h>

//---------------------------------------------------------------------
// INT8 Quantization
//
// Idea: Instead of save weight/activation with type float32 (4 bytes/element)
// map to int8 (1 byte/element) -> 4x smaller, 4x less bandwidth
//
// Formula:
//  x_q = clamp(round(x / scale), -128, 127)    [quantize]
//  x = x_q * scale                             [dequantize]
//
// scale = max(|x|) / 127   (per-tensor symmetric quantization)
//
// Wrong number: |x - dequant(quant(x))| <= scale/2   (quantization error)
// -------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Quantize kernel: float32 → int8
//
// Each thread processes 4 elements together (vector load) to optimize bandwidth
// float4 load: 1 instruction reads 4 floats instead of 4 individual instructions
// ---------------------------------------------------------------------------
__global__ void quantize_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 4 elements per step (vector load)
    int idx4 = idx * 4;
    if (idx4 + 3 < n) {
        // Load 4 floats in 1 instruction — increase memory throughput
        float4 in = reinterpret_cast<const float4*>(input)[idx];

        // Quantize: round → clamp → cast to int8
        // __float2int_rn: round-to-nearest, faster than roundf() because of hardware instruction
        output[idx4 + 0] = (int8_t)max(-128, min(127, __float2int_rn(in.x / scale)));
        output[idx4 + 1] = (int8_t)max(-128, min(127, __float2int_rn(in.y / scale)));
        output[idx4 + 2] = (int8_t)max(-128, min(127, __float2int_rn(in.z / scale)));
        output[idx4 + 3] = (int8_t)max(-128, min(127, __float2int_rn(in.w / scale)));
    } else {
        // Tail: Handle the remainder when n is not divisible by 4
        for (int i = idx4; i < n; i++) {
            output[i] = (int8_t)max(-128, min(127, __float2int_rn(input[i] / scale)));
        } 
    }
}

// -------------------------------------------------------------------------
// Dequantize kernel: int32 accumulator → float32
//
// After INT8 GEMM, accumulator is int32 (multiplication of two int8 may overflow int8)
// Dequantize: C_float = C_int32 * scale_A * scale_B
// -------------------------------------------------------------------------
__global__ void dequantize_kernel(
    const int32_t* __restrict__ input,
    float* __restrict__ output,
    float scale_A,
    float scale_B,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // total scale = scale_A * scale_B
    // Because: (a/scale_A) * (b/scale_B) = (a*b) / (scale_A*scale_B)
    output[idx] = (float)input[idx] * scale_A * scale_B;
}


// ------------------------------------------------------------------------
// INT8 GEMM kernel uses dp4a instruction
//
// dp4a (Dot Product of 4 elements, accumulate):
//  __dp4a(a, b, c) = c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//  With: a, b is int8x4 (packed into int32), c is int32 accumulator
//
// Throughput: 4x compared to the int8 scalar kernel, 4x compared to the float32 kernel.
// Similar to CUDA dp4a PTX instruction: dp4a.s32.s32
//
// Layout: A [M x K], B [K x N] (B is stored as column-major to optimize access)
// -----------------------------------------------------------------------
__global__ void int8_gemm_kernel(
    const int8_t* __restrict__ A,   // [M x K] row-major
    const int8_t* __restrict__ B,   // [N x K] — B transposed! column-major of B^T
    int32_t* __restrict__ C,        // [M x N] row-major
    int M, int N, int K
) {
    // Use tiled approach like float GEMM but with int8
    // TILE_K must be a multiple of 4 because dp4a processes 4 elements at a time.
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;  // phải là bội của 4

    __shared__ int8_t As[TILE_M][TILE_K + 4];   // +4 padding to avoid bank conflict
    __shared__ int8_t Bs[TILE_N][TILE_K + 4];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    int32_t c_val = 0;  // int32 accumulator — overflow when accumulator multiple int8

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile A
        int a_col = t * TILE_K + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0;

        // Load tile B (B transposed, so B[col][k] = B_original[k][col])
        int b_k = t * TILE_K + threadIdx.y;
        Bs[threadIdx.x][threadIdx.y] =
            (col < N && b_k < K) ? B[col * K + b_k] : 0;

        __syncthreads();

        // Compute dot product using dp4a — handles 4 elements per step
        // K must be a multiple of 4 for dp4a to be used effectively.
        for (int k = 0; k < TILE_K; k+= 4) {
            // Pack 4 int8 into 1 int32 to use dp4a
            int a_packed = *reinterpret_cast<const int*>(&As[threadIdx.y][k]);
            int b_packed = *reinterpret_cast<const int*>(&Bs[threadIdx.x][k]);

            // dp4a: c += dot(a[0:4], b[0:4]) in an instruction
            c_val = __dp4a(a_packed, b_packed, c_val);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}


// --------------------------------------------------------------------
// Host lauchers
// --------------------------------------------------------------------
void launch_quantize(
    const float* d_input,
    int8_t* d_output,
    float scale,
    int n
) {
    // 256 threads, each thread handles 4 elements → 1024 elements/block
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    quantize_kernel<<<blocks, threads>>>(d_input, d_output, scale, n);
}

void launch_dequantize(
    const int32_t* d_input,
    float*         d_output,
    float scale_A,
    float scale_B,
    int n
) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    dequantize_kernel<<<blocks, threads>>>(d_input, d_output, scale_A, scale_B, n);
}

void launch_int8_gemm(
    const int8_t* d_A,
    const int8_t* d_B,
    int32_t*      d_C,
    int M, int N, int K
) {
    // B must be transposed before call this kernel
    // (Handle in the preprocessing step, not in kernel to avoid overhead)
    dim3 block(16, 16);
    dim3 grid(
        (N + 15) / 16,
        (M + 15) / 16
    );
    int8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}