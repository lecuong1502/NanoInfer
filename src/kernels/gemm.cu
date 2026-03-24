#include "gemm.h"

#define TILE_SIZE 16

// -----------------------------------------------------------------
// Tiled GEMM (General Matrix Multiplication) with TILE_SIZE = 16
//
// Each thread block computes 1 [TILE_SIZE * TILE_SIZE] tile of matrix C.
// Instead of each block that reads from global memory (HBM) for each multiplication
// All blocks load 1 tile of A and B into shared memory
// Then all threads of block read from shared memory (~100x faster tgab HBM)
//
// Complexity: O(M*N*K) but the number of HBM accesses decreases from O(M*N*K) to O(M*N*K/TILE_SIZE)
// -----------------------------------------------------------------

// __restrict__ is a pointer type qualifier keyword: tells the compiler (nvcc) that a memory region 
// pointed to by that pointer is not being pointed to by any other pointer in that scope (pointer aliasing).
// This allows for optimized data loading, reduced global memory access, and improved kernel performance.

__global__ void gemm_tiled_kernel(
    const float* __restrict__ A, // [M x K] input matrix A
    const float* __restrict__ B, // [M x K] input matrix A
    float* __restrict__ C,       // [M x N] output matrix C
    int M, int N, int K
) {
    // Bank: Shared Memory on Nvidia GPU is separated into 32 banks with unique bandwidth.
    // Bank conflict: occurs when multiple threads on the same path attempt to access different memory addresses within the same shared memory "bank".
    // Shared memory tiles: +1 padding to avoid bank conflict while reading the column of As
    // If no padding, then 16 threads read the same bank -> 16-way conflict -> 16x slower
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Output location of thread in matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulator — hold in register during the loop, not write into memory
    float c_val = 0.0f;

    // Number of tiles to traverse in the K-axis
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A to shared memory
        // Each thread load only 1 element: As[threadIdx.y][threadIdx.x]
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;    // Zero-padding if out of bounds
        }

        // Load tile of B to shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Barrier: wait all threads in block to load completely then calculate
        __syncthreads();

        // Compute dot product for this tile
        // Read from shared memory (~4 cycles) instead of HBM (~600 cycles)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Barrier: wait all threads to compute completely before load a new tile
        __syncthreads();
    }

    // Write thr results to HBM
    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

// -------------------------------------------------------------------
// Tiled GEMM kernel (TILE_SIZE = 32)
// Bigger TILE_SIZE -> each tile contains more data -> fewer HBM loads
// -------------------------------------------------------------------
#define TILE_SIZE_32 32

__global__ void gemm_tiled_32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE_32][TILE_SIZE_32 + 1];  // +1 padding to avoid bank conflict
    __shared__ float Bs[TILE_SIZE_32][TILE_SIZE_32];

    int row = blockIdx.y * TILE_SIZE_32 + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_32 + threadIdx.x;
    float c_val = 0.0f;

    int num_tiles = (K + TILE_SIZE_32 - 1) / TILE_SIZE_32;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE_32 + threadIdx.x;
        int b_row = t * TILE_SIZE_32 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

// -----------------------------------------------------------------
// Hot launcher - select the appropriate TILE_SIZE based on the matrix size.
// -----------------------------------------------------------------
void launch_gemm_tiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K
) {
    // Use TILE_SIZE = 32 for big matrix (>= 512 on 3 dimensions)
    // Because register pressure of 32x32 is too big for small matrix
    if (M >= 512 && N >= 512 && K >= 512) {
        dim3 block(TILE_SIZE_32, TILE_SIZE_32);
        dim3 grid(
            (N + TILE_SIZE_32 - 1) / TILE_SIZE_32,
            (M + TILE_SIZE_32 - 1) / TILE_SIZE_32
        );
        gemm_tiled_32_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE
        );
        gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
}