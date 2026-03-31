#pragma once
#include <string>
#include <vector>
#include "kvcache.h"

// ---------------------------------------------------------------------------
// Precision modes for model weights and activations
// ---------------------------------------------------------------------------
enum class Precision { FP32, FP16, INT8 };

// ---------------------------------------------------------------------------
// GPT2Model
//
// Implements the GPT-2 transformer forward pass using custom CUDA kernels:
//   - Fused LayerNorm + Linear (layers/linear.h)
//   - Flash Attention v1      (kernels/attention.h)
//   - Token + positional embedding (layers/embedding.h)
//
// Architecture (GPT-2 small, 124M parameters):
//   vocab_size=50257, d_model=768, n_layers=12,
//   n_heads=12, d_head=64, d_ffn=3072, max_seq_len=1024
// ---------------------------------------------------------------------------
class GPT2Model {
public:
    // Load model weights from model_path directory.
    // precision: compute precision for activations (FP32 / FP16 / INT8).
    // max_seq_len: maximum sequence length supported (caps KV-cache size).
    GPT2Model(
        const std::string& model_path,
        Precision          precision   = Precision::FP32,
        int                max_seq_len = 1024
    );
    ~GPT2Model();

    std::vector<float> forward(
        const std::vector<int>& input_ids,
        PagedKVCache&           cache
    );

    // Tokenize plain text → token id sequence (BPE).
    std::vector<int> tokenize(const std::string& text) const;

    // Reconstruct text from a sequence of token ids.
    std::string detokenize(const std::vector<int>& ids) const;

    // ---- Model metadata ----
    int eos_token_id() const;  // GPT-2: 50256 (<|endoftext|>)
    int vocab_size()   const;  // 50257
    int num_layers()   const;  // 12
    int num_heads()    const;  // 12
    int head_dim()     const;  // 64
    int d_model()      const;  // 768

private:
    struct Impl;
    Impl* impl_;
};