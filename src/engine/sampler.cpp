#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// -------------------------------------------------------------------
// Sampler — top-k / top-p (nucleus) sampling for autoregressive generation
//
// Pipeline for each steps:
//      1. Apply temperature scaling: logits /= temperature
//      2. Top-k filter: zero out all but the k highest logits
//      3. Softmax: convert logits to probabilities
//      4. Top-p filter: keep smallest prefix of tokens whose cumulative prob >= p
//      5. Sample: draw one token from the filtered distribution
// -------------------------------------------------------------------

Sampler::Sampler(const SamplerConfig& config) 
    : cfg_(config)
{
    // LCG seed — simple but sufficient for sampling (not crypto-quality)
    rng_state_ = config.seed ^ 0xdeadbeef;
}

// Fast LCG random float in [0, 1)
static float lcg_float(uint32_t& state) {
    state = state * 1664525u + 1013904223u;  // Numerical Recipes LCG constants
    return (float)(state >> 8) / (float)(1 << 24);
}

int Sampler::sample(const std::vector<float>& logits) {
    int V = (int)logits.size();
    if (V == 0) throw std::runtime_error("sampler: empty logits");

    // --- Greedy shortcut: temperature=0 means argmax ---
    if (cfg_.temperature <= 0.0f) {
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // --- Step 1: Temperature scaling ---
    // Higher temperature → flatter distribution → more random output
    // Lower temperature → sharper distribution → more deterministic output
    std::vector<float>scaled(V);
    for (int i = 0; i < V; i++)
        scaled[i] = logits[i] / cfg_.temperature;

    // --- Step 2: Top-k filtering ---
    // Zero out all logits except the top-k — prevents very unlikely tokens
    if (cfg_.top_k > 0 && cfg_.top_k < V) {
        // Find the k-th largest value via partial sort
        std::vector<float> sorted_vals(scaled);
        std::nth_element(sorted_vals.begin(),
                         sorted_vals.begin() + (V - cfg_.top_k),
                         sorted_vals.end());
        float kth_val = sorted_vals[V - cfg_.top_k];

        // Zero out everything below the cutoff
        for (int i = 0; i < V; i++)
            if (scaled[i] < kth_val)
                scaled[i] = -1e9f;  // effectively -inf after softmax
    }

    // --- Step 3: Stable softmax ---
    // Subtract max for numerical stability (avoids exp overflow)
    float max_val = *std::max_element(scaled.begin(), scaled.end());
    std::vector<float> probs(V);
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs[i] = std::exp(scaled[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < V; i++) probs[i] /= sum;

    // --- Step 4: Top-p (nucleus) filtering ---
    // Sort tokens by probability descending, keep smallest set with cumsum >= p
    // This adaptively narrows the sampling pool based on probability mass
    if (cfg_.top_p < 1.0f) {
        // Get indices sorted by probability (descending)
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });

        // Find cutoff: smallest prefix with cumulative prob >= top_p
        float cumsum = 0.0f;
        int   cutoff = V;
        for (int i = 0; i < V; i++) {
            cumsum += probs[idx[i]];
            if (cumsum >= cfg_.top_p) { cutoff = i + 1; break; }
        }

        // Zero out tokens outside the nucleus
        for (int i = cutoff; i < V; i++) probs[idx[i]] = 0.0f;

        // Renormalize after zeroing
        sum = 0.0f;
        for (int i = 0; i < V; i++) sum += probs[i];
        if (sum > 0.0f)
            for (int i = 0; i < V; i++) probs[i] /= sum;
    }

    // --- Step 5: Sample from filtered distribution ---
    // Linear scan through CDF — O(V) but simple and correct
    float u = lcg_float(rng_state_);   // uniform [0, 1)
    float cdf = 0.0f;
    for (int i = 0; i < V; i++) {
        cdf += probs[i];
        if (u < cdf) return i;
    }

    // Fallback: floating point rounding can leave cdf slightly < 1
    // Return last non-zero token
    for (int i = V - 1; i >= 0; i--)
        if (probs[i] > 0.0f) return i;

    return V - 1;
}