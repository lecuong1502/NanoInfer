#include "kvcache.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <cassert>

using namespace std;
// ----------------------------------------------------------
// TurboQuant KV Cache — C++ Implementation
//
// Paper: "TurboQuant: Online Vector Quantization with Near-optimal
//         Distortion Rate", Zandieh et al., April 2025 (arXiv:2504.19874)
//
// Core idea (Section 3 of paper):
//
//  TurboQuantMSE (Algorithm 1):
//      1. Rotate input vector x by random orthogonal matrix Π
//          → rotated coords follow Beta distribution (Lemma 1)
//      2. Quantize each coords to nearest Lloyd-Max centroid
//          → near-optimal MSE: Dmse ≤ (√3π/2) * 4^(-b)
//  `   3. Dequant: lookup centroids, rotate back by Π^T
//
//  TurboQuantPROD (Algorithm 2) — for KV cache attention queries:
//      1. Apply TurboQuantMSE with (b - 1) bits
//      2. Compute residual r = x - dequant(quant(x))
//      3. Apply QJL (1-bit): qjl = sign(S · r), store ‖r‖₂
//      4. Dequant: x̃ = x̃_mse + (√π/2 / d) * ‖r‖ * S^T * qjl
//
//  Properties:
//      - Unbiased: E[⟨y, x̃⟩] = ⟨y, x⟩  (crucial for attention correctness)
//      - Distortion: Dprod ≤ (√3π/2) * ‖y‖² / d * 4^(-b)
//      - At 3.5 bits: matches full-precision quality (Table 1 of paper)
//      - Online (no calibration data needed) - suitable for KV cache
// --------------------------------------------------------


// --------------------------------------------------------
// Lloyd-Max codebooks precomputed for Beta/Normal distribution
//
// For moderate-to-high head_dim (d ≥ 64), each coord of the rotated vector
// approximates N(0, 1/d). We precompute Lloyd-Max centroids for this
// distribution at bit-widths 2, 3, 4 (Table in Section 3.1 of paper)
//
// These match the paper's reported distortions:
//      b=2: Dmse ≈ 0.117,  b=3: Dmse ≈ 0.03,  b=4: Dmse ≈ 0.009
//
// Centroids are stored normalized (assume variance = 1, scale by 1/sqrt(d)
// at quantize time to match the actual N(0, 1/d) distribution)
// --------------------------------------------------------

// Centroids for N(0, 1) — scale by 1/sqrt(d) when quantizing N(0, 1/d) coords
static const float CENTROIDS_2BIT[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

static const float CENTROIDS_3BIT[8] = {
    -2.1520f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1520f
};

static const float CENTROIDS_4BIT[16] = {
    -2.7326f, -2.0690f, -1.5916f, -1.1890f,
    -0.8345f, -0.5188f, -0.2360f,  0.0000f,
     0.2360f,  0.5188f,  0.8345f,  1.1890f,
     1.5916f,  2.0690f,  2.7326f,  3.1685f
};

static const float* get_centroids(int bits) {
    switch (bits) {
        case 2: return CENTROIDS_2BIT;
        case 3: return CENTROIDS_3BIT;
        case 4: return CENTROIDS_4BIT;
        default: throw invalid_argument("bits must be 2, 3, or 4");
    }
}

static int num_centroids(int bits) { return 1 << bits; }

// -----------------------------------------------------
// TurboQuantMSE: quantize one d-dimensional vector (CPU path)
//
// Algorithm 1 from the paper:
//   y     = Π · x              (random rotation)
//   idx_j = argmin_k |y_j - c_k|  for each coord j
//   x̃    = Π^T · [c_{idx_j}]   (dequant: lookup + rotate back)
// -----------------------------------------------------
static void turboquant_mse_quantize(
    const float* x,          // input vector [d]
    const float* Pi,         // rotation matrix [d x d], row-major
    uint8_t* idx,            // output indices [d], b bits each
    int d,
    int bits
) {
    const float* centroids = get_centroids(bits);
    int K = num_centroids(bits);
    float inv_sqrtd = 1.0f / sqrt((float)d);

    // y = Π · x
    vector<float> y(d, 0.0f);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            y[i] += Pi[i * d + j] * x[j];

    // Quantize each coord: nearest centroid (scaled for N(0, 1/d))
    for (int j = 0; j < d; j++) {
        float yj   = y[j] / inv_sqrtd;  // rescale to unit variance
        float best = fabs(yj - centroids[0]);
        idx[j] = 0;
        for (int k = 1; k < K; k++) {
            float dist = std::fabs(yj - centroids[k]);
            if (dist < best) { best = dist; idx[j] = (uint8_t)k; }
        }
    }
}

static void turboquant_mse_dequantize(
    const uint8_t* idx,     // quantized indices [d]
    const float* Pi,      // rotation matrix [d x d]
    float* x_hat,   // output reconstructed vector [d]
    int d,
    int bits
) {
    const float* centroids = get_centroids(bits);
    float inv_sqrtd = 1.0f / std::sqrt((float)d);

    // y̌_j = c_{idx_j} * (1/sqrt(d))  (undo the unit-variance scaling)
    vector<float> y_hat(d);
    for (int j = 0; j < d; j++) 
        y_hat[j] = centroids[idx[j]] * inv_sqrtd;

    // x̃ = Π^T · ŷ  (Π is orthogonal → Π^T = Π^{-1})
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d; j++) 
            acc += Pi[j * d + i] * y_hat[j];  // Pi^T[i,j] = Pi[j,i]
        x_hat[i] = acc;
    }
}


// -------------------------------------------------------------
// TurboQuantPROD: Algorithm 2 from the paper
//
// Stage 1: Apply TurboQuantMSE with (b-1) bits
// Stage 2: Compute residual r = x - x̃_mse
// Stage 3: QJL on residual — qjl = sign(S · r), store ‖r‖₂
//
// Dequant:
//      x̃ = x̃_mse + (√(π/2) / d) * ‖r‖ * S^T * qjl
//
// This guarantees E[⟨y, x̃⟩] = ⟨y, x⟩ for any query y (Theorem 2).
// ------------------------------------------------------------

// Quantized representation of one KV vector in PROD mode
struct TQProdEntry {
    vector<uint8_t> mse_idx;    // (b-1)-bit MSE indices, d elements
    vector<int8_t>  qjl_signs;  // 1-bit QJL signs, d elements ∈ {-1,+1}
    float r_norm;               // ‖r‖₂ — scalar needed for dequant
};

static TQProdEntry turboquant_prod_quantize(
    const float* x,
    const float* Pi,    // rotation matrix for MSE stage [d x d]
    const float* S,     // random Gaussian projection matrix [d x d]
    int d,
    int bits            // effective bit-width (MSE uses bits-1, QJL uses 1)
) {
    TQProdEntry entry;
    entry.mse_idx.resize(d);
    entry.qjl_signs.resize(d);

    // Stage 1: TurboQuantMSE with (bits-1) bits
    turboquant_mse_quantize(x, Pi, entry.mse_idx.data(), d, bits - 1);

    // Compute x̃_mse (dequantized MSE estimate)
    vector<float> x_hat_mse(d);
    turboquant_mse_dequantize(entry.mse_idx.data(), Pi, x_hat_mse.data(), d, bits - 1);

    // Stage 2: residual r = x - x̃_mse
    vector<float> r(d);
    float r_norm_sq = 0.0f;
    for (int i = 0; i < d; i++) {
        r[i] = x[i] - x_hat_mse[i];
        r_norm_sq += r[i] * r[i];
    }
    entry.r_norm = sqrt(r_norm_sq);

    // Stage 3: QJL — qjl_j = sign(S_j · r)
    // S is [d x d] random Gaussian, S_j is j-th row
    for (int j = 0; j < d; j++) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++)
            dot += S[j * d + k] * r[k];
        entry.qjl_signs[j] = (dot >= 0.0f) ? 1 : -1;
    }

    return entry;
}

static void turboquant_prod_dequantize(
    const TQProdEntry& entry,
    const float* Pi,
    const float* S,
    float* x_out,
    int d,
    int bits
) {
    // x̃_mse from MSE dequant
    vector<float> x_hat_mse(d);
    turboquant_mse_dequantize(entry.mse_idx.data(), Pi, x_hat_mse.data(), d, bits - 1);

    // QJL contribution: (√(π/2) / d) * ‖r‖ * S^T · qjl
    // S^T[i,j] = S[j,i]
    float scale = sqrt(M_PI / 2.0f) / (float)d * entry.r_norm;
    for (int i = 0; i < d; i++) {
        float qjl_contrib = 0.0f;
        for (int j = 0; j < d; j++)
            qjl_contrib += S[j * d + i] * (float)entry.qjl_signs[j];
        x_out[i] = x_hat_mse[i] + scale * qjl_contrib;
    }
}


// ------------------------------------------
// PagedKVCache:: Impl - internal state
// ------------------------------------------
struct PagedKVCache::Impl {
    int num_layers;
    int num_heads;
    int head_dim;
    int max_pages;
    int page_size;
    int bits;
    KVQuantMode mode;
    
    int seq_len;    // current number of stored tokens

    // Random rotation matrix Π [head_dim x head_dim] (shared across layers)
    // Generated once at construction — data-oblivious (key TurboQuant property)
    vector<float> Pi;

    // Random Gaussian projection S [head_dim x head_dim] for QJL stage
    // Only allocated in PROD mode
    vector<float> S;

    // Quantized KV storage: [num_layers][num_heads][max_tokens]
    // Each entry is a TQProdEntry (PROD mode) or raw FP32 (NONE mode)
    vector<vector<vector<TQProdEntry>>> kv_store_k;
    vector<vector<vector<TQProdEntry>>> kv_store_v;

    // FP32 fallback storage (NONE mode)
    // [num_layers x num_heads x max_tokens x head_dim]
    vector<float> fp32_k;
    vector<float> fp32_v;

    // Device-side scratch buffers for retrieve() output
    float* d_key_scratch   = nullptr;
    float* d_value_scratch = nullptr;
    int scratch_tokens = 0;

    Impl(int nl, int nh, int hd, int mp, int ps, int b, KVQuantMode m)
        : num_layers(nl), num_heads(nh), head_dim(hd)
        , max_pages(mp), page_size(ps), bits(b), mode(m), seq_len(0)
    {
        int max_tokens = max_pages * page_size;

        // Generate random rotation matrix Π via QR decomposition of random Gaussian
        // "We can generate Π by applying QR decomposition on a random matrix
        //  with i.i.d. Normal entries." — Section 3.1
        mt19937 rng(42);    // fixed seed → deterministic, reproducible
        normal_distribution<float> normal(0.0f, 1.0f);

        Pi.resize(hd * hd);
        for (auto& v : Pi) v = normal(rng);
        gram_schmidt(Pi.data(), hd);  // orthogonalize in-place

        if (mode == KVQuantMode::PROD) {
            // S: random Gaussian projection for QJL
            // "S ∈ R^{d×d} with i.i.d. entries sampled from N(0,1)" — Definition 1
            S.resize(hd * hd);
            for (auto& v : S) v = normal(rng);
            // Note: S does NOT need to be orthogonal — raw Gaussian is correct

            // Allocate quantized storage
            kv_store_k.assign(nl, vector<vector<TQProdEntry>>(
                nh, vector<TQProdEntry>(max_tokens)));
            kv_store_v.assign(nl, vector<vector<TQProdEntry>>(
                nh, vector<TQProdEntry>(max_tokens)));
        } else if (mode == KVQuantMode::MSE) {
            kv_store_k.assign(nl, vector<vector<TQProdEntry>>(
                nh, vector<TQProdEntry>(max_tokens)));
            kv_store_v.assign(nl, vector<vector<TQProdEntry>>(
                nh, vector<TQProdEntry>(max_tokens)));
        } else {
            // NONE: plain FP32 storage
            size_t total = (size_t)nl * nh * max_tokens * hd;
            fp32_k.resize(total, 0.0f);
            fp32_v.resize(total, 0.0f);
        }
    }

    ~Impl() {
        if (d_key_scratch) cudaFree(d_key_scratch);
        if (d_value_scratch) cudaFree(d_value_scratch);
    }

    // Gram-Schmidt orthogonalization to produce rotation matrix Π
    // Simple column-by-column QR: sufficient for d ≤ 256 (GPT-2 head_dim=64)
    static void gram_schmidt(float* A, int d) {
        for (int j = 0; j < d; j++) {
            // Orthogonalize col j against all previous cols
            for (int k = 0; k < j; k++) {
                float dot = 0.0f;
                for (int i = 0; i < d; i++)
                    dot += A[i * d + j] * A[i * d + k];
                for (int i = 0; i < d; i++)
                    A[i * d + j] -= dot * A[i * d + k];
            }
            // Normalize col j
            float norm = 0.0f;
            for (int i = 0; i < d; i++)
                norm += A[i * d + j] * A[i * d + j];
            norm = sqrt(norm);
            if (norm > 1e-8f)
                for (int i = 0; i < d; i++)
                    A[i * d + j] /= norm;
        }
    }

    // Ensure scratch buffers are large enough for seq_len tokens
    void ensure_scratch(int tokens) {
        if (tokens <= scratch_tokens) return;
        if (d_key_scratch) cudaFree(d_key_scratch);
        if (d_value_scratch) cudaFree(d_value_scratch);
        size_t bytes = (size_t)tokens * head_dim * sizeof(float);
        cudaMalloc(&d_key_scratch, bytes);
        cudaMalloc(&d_value_scratch, bytes);
        scratch_tokens = tokens;
    }
};


// ---------------------------------------
// PagedKVCache public interface
// ---------------------------------------

PagedKVCache::PagedKVCache(
    int num_layers, int num_heads, int head_dim,
    int max_pages,  int page_size, int bits, KVQuantMode mode)
    : impl_(new Impl(num_layers, num_heads, head_dim,
                    max_pages, page_size, bits, mode))
{}

PagedKVCache::~PagedKVCache() { delete impl_; }

int PagedKVCache::current_seq_len() const { return impl_->seq_len; }

void PagedKVCache::reset() {
    impl_->seq_len = 0;
    // Quantized entries are overwritten on next append — no need to clear
}

// append: quantize and store one new token's K/V vectors
void PagedKVCache::append(
    int layer, int head,
    const float* d_key, const float* d_value
) {
    int d = impl_->head_dim;
    int t = impl_->seq_len;     // Slot to write into

    // Copy K and V from device to host for CPU-side TurboQuant
    // (TurboQuant's rotation and quantization are CPU ops here;
    //  a production implementation would use a CUDA kernel for Π·x)
    vector<float> h_key(d), h_val(d);
    cudaMemcpy(h_key.data(), d_key,   d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val.data(), d_value, d * sizeof(float), cudaMemcpyDeviceToHost);

    if (impl_->mode == KVQuantMode::PROD) {
        // TurboQuantPROD — Algorithm 2
        impl_->kv_store_k[layer][head][t] = turboquant_prod_quantize(
            h_key.data(), impl_->Pi.data(), impl_->S.data(), d, impl_->bits);
        impl_->kv_store_v[layer][head][t] = turboquant_prod_quantize(
            h_val.data(), impl_->Pi.data(), impl_->S.data(), d, impl_->bits);

    } else if (impl_->mode == KVQuantMode::MSE) {
        // TurboQuantMSE — Algorithm 1
        // Reuse TQProdEntry struct (only mse_idx is populated, qjl/r_norm unused)
        TQProdEntry ke, ve;
        ke.mse_idx.resize(d); ve.mse_idx.resize(d);
        turboquant_mse_quantize(h_key.data(), impl_->Pi.data(),
                                ke.mse_idx.data(), d, impl_->bits);
        turboquant_mse_quantize(h_val.data(), impl_->Pi.data(),
                                ve.mse_idx.data(), d, impl_->bits);
        impl_->kv_store_k[layer][head][t] = move(ke);
        impl_->kv_store_v[layer][head][t] = move(ve);

    } else {
        // None: FP32 Storage
        size_t stride = (size_t)impl_->num_heads * impl_->max_pages
                        * impl_->page_size * d;
        size_t offset = (size_t)layer * stride
                      + (size_t)head  * impl_->max_pages * impl_->page_size * d
                      + (size_t)t     * d;
        memcpy(impl_->fp32_k.data() + offset, h_key.data(), d * sizeof(float));
        memcpy(impl_->fp32_v.data() + offset, h_val.data(), d * sizeof(float));
    }

    // Only increment once per full layer pass (caller increments after all layers)
    // Design choice: caller controls seq_len to allow per-layer flexibility
}

// retrieve: dequantize all stored K/V for one head and upload to device
void PagedKVCache::retrieve(
    int layer, int head, int seq_len,
    float* d_key_out, float* d_value_out) const
{
    int d = impl_->head_dim;
    impl_->ensure_scratch(seq_len);

    std::vector<float> h_key(seq_len * d), h_val(seq_len * d);

    for (int t = 0; t < seq_len; t++) {
        float* krow = h_key.data() + t * d;
        float* vrow = h_val.data() + t * d;

        if (impl_->mode == KVQuantMode::PROD) {
            turboquant_prod_dequantize(
                impl_->kv_store_k[layer][head][t],
                impl_->Pi.data(), impl_->S.data(), krow, d, impl_->bits);
            turboquant_prod_dequantize(
                impl_->kv_store_v[layer][head][t],
                impl_->Pi.data(), impl_->S.data(), vrow, d, impl_->bits);

        } else if (impl_->mode == KVQuantMode::MSE) {
            turboquant_mse_dequantize(
                impl_->kv_store_k[layer][head][t].mse_idx.data(),
                impl_->Pi.data(), krow, d, impl_->bits);
            turboquant_mse_dequantize(
                impl_->kv_store_v[layer][head][t].mse_idx.data(),
                impl_->Pi.data(), vrow, d, impl_->bits);

        } else {
            size_t stride = (size_t)impl_->num_heads * impl_->max_pages
                            * impl_->page_size * d;
            size_t base = (size_t)layer * stride
                        + (size_t)head  * impl_->max_pages * impl_->page_size * d
                        + (size_t)t     * d;
            memcpy(krow, impl_->fp32_k.data() + base, d * sizeof(float));
            memcpy(vrow, impl_->fp32_v.data() + base, d * sizeof(float));
        }
    }

    // Upload reconstructed vectors to device
    cudaMemcpy(d_key_out,   h_key.data(), seq_len * d * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_out, h_val.data(), seq_len * d * sizeof(float),
               cudaMemcpyHostToDevice);
}

float* PagedKVCache::key_ptr(int layer) const {
    return impl_->d_key_scratch;
}

float* PagedKVCache::value_ptr(int layer) const {
    return impl_->d_value_scratch;
}