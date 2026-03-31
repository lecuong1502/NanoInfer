#pragma once
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// SamplerConfig — controls the token sampling strategy
//
// temperature: scales logits before softmax.
//   0.0  → greedy (argmax, deterministic)
//   1.0  → standard sampling
//   >1.0 → more random output
//
// top_k: keep only the k highest-probability tokens before sampling.
//   0    → disabled (consider all tokens)
//   50   → typical value for GPT-2 generation
//
// top_p: nucleus sampling — keep the smallest set of tokens whose
//   cumulative probability ≥ top_p, then sample from that set.
//   1.0  → disabled
//   0.95 → typical value (filters very low probability tail)
//
// seed: RNG seed for reproducibility.
//   Fixed seed → deterministic output for the same input.
// ---------------------------------------------------------------------------
struct SamplerConfig {
    float    temperature = 1.0f;
    float    top_p       = 0.95f;
    int      top_k       = 50;
    uint32_t seed        = 42;
};

// ---------------------------------------------------------------------------
// Sampler
//
// Converts raw logits [vocab_size] to a single sampled token id.
//
// Pipeline per step:
//   1. Temperature scaling  — logits /= temperature
//   2. Top-k filter         — zero out all but top-k logits
//   3. Softmax              — logits → probabilities (numerically stable)
//   4. Top-p (nucleus) filter — keep cumulative mass ≥ top_p, renormalize
//   5. Sample               — draw one token from filtered distribution
//
// Uses a fast LCG (linear congruential generator) for sampling —
// no dependency on std::mt19937 state across calls, deterministic given seed.
// ---------------------------------------------------------------------------
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config);

    int sample(const std::vector<float>& logits);

private:
    SamplerConfig cfg_;
    uint32_t      rng_state_;  // LCG state, updated on each sample() call
};