#pragma once
#include <cuda_runtime.h>

// Paged KV-cache: pre-allocates a fixed pool of pages,
// assigns them to sequence positions on demand.
class PagedKVCache {
public:
    PagedKVCache(int num_layers,
                 int num_heads,
                 int head_dim,
                 int max_pages);
    ~PagedKVCache();

    // Returns device pointers to K and V for a given layer
    float* key_ptr(int layer) const;
    float* value_ptr(int layer) const;

    int current_seq_len() const;
    void reset();   // clear cache between requests

private:
    struct Impl;
    Impl* impl_;
};