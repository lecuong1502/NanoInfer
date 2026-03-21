#pragma once
#include <vector>
#include <cstdint>

struct SamplerConfig {
    float    temperature = 1.0f;
    float    top_p       = 0.95f;
    int      top_k       = 50;
    uint32_t seed        = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config);

    // Sample one token id from logits [vocab_size]
    int sample(const std::vector<float>& logits);

private:
    SamplerConfig cfg_;
    uint32_t      rng_state_;
};