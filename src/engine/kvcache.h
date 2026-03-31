#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

enum class KVQuantMode { NONE, MSE, PROD };

// Quantized storage entry for one KV vector (used internally by Impl)
struct TQProdEntry {
    std::vector<uint8_t> mse_idx;    // Lloyd-Max indices, one per coord
    std::vector<int8_t>  qjl_signs;  // QJL sign bits ∈ {-1, +1}, PROD only
    float                r_norm;     // ‖residual‖₂ scalar,       PROD only
};

class PagedKVCache {
public:
    PagedKVCache(
        int         num_layers,
        int         num_heads,
        int         head_dim,
        int         max_pages,
        int         page_size  = 16,
        int         bits       = 4,
        KVQuantMode mode       = KVQuantMode::PROD
    );
    ~PagedKVCache();

    void append(
        int          layer,
        int          head,
        const float* d_key,
        const float* d_value
    );
    
    void retrieve(
        int    layer,
        int    head,
        int    seq_len,
        float* d_key_out,
        float* d_value_out
    ) const;

    float* key_ptr  (int layer) const;
    float* value_ptr(int layer) const;

    // Number of tokens currently stored (incremented externally by the model).
    int  current_seq_len() const;

    // Reset the cache between requests — clears seq_len, reuses memory.
    void reset();

private:
    struct Impl;
    Impl* impl_;
};