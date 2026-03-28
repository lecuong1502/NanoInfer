#include "embedding.h"

// -------------------------------------------------------------
// Token Embedding
//
// Embedding Lookup: For each token ID, copy the corresponding row from the embedding table.
//
//  output[i] = weight[token_ids[i]]    (copy d_model floats)
//
// This is the first step of GPT-2 forward pass — transfer token ids (integers)
// to dense vectors that transformer can handle
//
// Embedding table weight: [vocab_size x d_model]
//  GPT-2 small: vocab_size=50257, d_model=768 → 50257*768*4 = 154 MB
//
// Access pattern: random access to embedding table (each different token id)
// → No cache-friendly, but not prevent because of depending on input
// ------------------------------------------------------------
__global__ void token_embedding_kernel(
    const int* __restrict__ token_ids,  // [seq_len]
    const float* __restrict__ weight,   // [vocab_size x d_model]
    float* __restrict__ output,         // [seq_len x d_model]
    int seq_len, int d_model
){
    // Each block handles 1 token
    int token_pos = blockIdx.x;
    if (token_pos >= seq_len) return;

    int token_id = token_ids[token_pos];

    // Pointer to corresponding row in embedding table
    const float* embed_row = weight + token_id * d_model;
    float* out_row   = output + token_pos * d_model;

    // Copy d_model floats — use vectorized load (float4) to optimize bandwidth
    // float4: 4 floats in 1 transaction 128-bit instead of 4 transactions 32-bit
    int d4 = d_model / 4;   // number of float4 chunks
    for (int i = threadIdx.x; i < d4; i += blockDim.x) {
        reinterpret_cast<float4*>(out_row)[i] = 
            reinterpret_cast<const float4*>(embed_row)[i];
    }
    
    // Tail: Handle the remainder if d_model is not divisible by 4
    // GPT-2: d_model=768=192*4 → there is no tail, but true to all models
    int tail_start = d4 * 4;
    for (int i = tail_start + threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = embed_row[i];
    }
}

// --------------------------------------------------------
// Positional Embedding
//
// Add positional embedding to token embedding:
//  output[i] += pos_weight[position_offset + i]
//
// GPT-2 uses learned positional embeddings (not sinusoidal)
// → pos_weight is weight matrix that can be trained, size = [max_seq_len x d_model]
//
// position_offset: use when generating with KV-cache
//  - First step (prefill): offset = 0, seq_len = prompt length
//  - Next steps (decode): offset = current_seq_len, seq_len = 1
//  → Each new token take positional embedding in its position
// -------------------------------------------------------
__global__ void positional_embedding_kernel(
    const float* __restrict__ pos_weight,   // [max_seq_len x d_model]
    float* __restrict__ output,             // [seq_len x d_model] — modified in-place
    int seq_len, int d_model,
    int position_offset
) {
    int token_pos = blockIdx.x;
    if (token_pos >= seq_len) return;

    // Positional embedding for token in position (position_offset + token_pos)
    const float* pos_row = pos_weight + (position_offset + token_pos) * d_model;
    float* out_row = output + token_pos * d_model;

    // Add in-place: output[token_pos][i] += pos_weight[offset + token_pos][i]
    // Use float4 to increase throughput
    int d4 = d_model / 4;
    for (int i = threadIdx.x; i < d4; i += blockDim.x) {
        float4 o = reinterpret_cast<float4*>(out_row)[i];
        float4 pos = reinterpret_cast<const float4*>(pos_row)[i];
        // Add each component
        o.x += pos.x; o.y += pos.y;
        o.z += pos.z; o.w += pos.w;
        reinterpret_cast<float4*>(out_row)[i] = o;
    }

    int tail_start = d4 * 4;
    for (int i = tail_start + threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] += pos_row[i];
    }
}

// ---------------------------------------------------------
// Host lauchers
// ---------------------------------------------------------

void launch_token_embedding(
    const int* d_token_ids,
    const float* d_weight,
    float* d_output,
    int seq_len, int vocab_size, int d_model
) {
    // 1 block per token, 128 threads per block
    // Each thread copies d_model/128 floats (with float4: d_model/(128*4) chunks)
    int threads = 128;
    token_embedding_kernel<<<seq_len, threads>>>(
        d_token_ids, d_weight, d_output, seq_len, d_model);
}

void launch_positional_embedding(
    const float* d_pos_weight,
    float*       d_output,
    int seq_len, int d_model,
    int position_offset
) {
    int threads = 128;
    positional_embedding_kernel<<<seq_len, threads>>>(
        d_pos_weight, d_output, seq_len, d_model, position_offset);
}