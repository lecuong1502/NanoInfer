#pragma once
#include <cuda_runtime.h>

// Row-wise naive softmax — 3 passes, NOT numerically stable (may overflow on large values)
// Only used in benchmarks for comparison. Not recommended for production.
// input/output: [rows x cols]
void launch_softmax_naive(
    const float* d_input,
    float*       d_output,
    int rows, int cols
);

// Row-wise safe softmax — 3 passes, numerically stable (subtracts max before exp)
// input/output: [rows x cols]
void launch_softmax_safe(
    const float* d_input,
    float*       d_output,
    int rows, int cols
);

// Row-wise online softmax (single-pass, numerically stable) ← NanoInfer default
// input/output: [rows x cols]
void launch_softmax_online(
    const float* d_input,
    float*       d_output,
    int rows, int cols
);