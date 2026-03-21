#pragma once
#include <cuda_runtime.h>

// Row-wise online softmax (single-pass, numerically stable)
// input/output: [rows x cols]
void launch_softmax_online(
    const float* d_input,
    float* d_output,
    int rows, int cols
);