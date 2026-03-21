#pragma once
#include <string>
#include <vector>
#include "kvcache.h"

enum class Precision { FP32, FP16, INT8 };

class GPT2Model {
public:
    GPT2Model(const std::string& model_path,
              Precision precision,
              int max_seq_len);
    ~GPT2Model();

    // Forward pass - return logits [vocab_size]
    std::vector<float> forward(const std::vector<int>& input_ids,
                               PagedKVCache& cache);
 
    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& ids) const;

    int eos_token_id() const;
    int vocab_size() const;
    int num_layers() const;
    int num_heads() const;
    int head_dim() const;
    int d_model() const;

private:
    struct Impl;
    Impl* impl_;
};