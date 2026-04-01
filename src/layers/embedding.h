#pragma once
#include <cuda_runtime.h>

// Token embedding lookup: output[i] = weight[token_ids[i]]
// token_ids: [seq_len]     (int32)
// weight: [vocab_size x d_model]
// output: [seq_lem x d_model]
void launch_token_embedding(
    const int*   d_token_ids,
    const float* d_weight,
    float*       d_output,
    int seq_len, int vocab_size, int d_model
);

// Positional embedding: adds learned position vectors to token embeddings
// output[i] += pos_weight[position_offset + i]
// pos_weight: [max_seq_len x d_model]
void launch_positional_embedding(
    const float* d_pos_weight,
    float* d_output,     // modified in-place
    int seq_len, int d_model,
    int position_offset = 0
);  // for KV-cache: offset into position table