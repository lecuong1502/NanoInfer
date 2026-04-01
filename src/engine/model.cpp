#include "model.h"
#include "kvcache.h"
#include "../layers/linear.h"
#include "../layers/layernorm.h"
#include "../layers/embedding.h"
#include "../layers/activation.h"
#include "../kernels/attention.h"
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// GPT-2 Model — forward pass assembly
//
// Architecture (GPT-2 small, 124M params):
//   vocab_size = 50257, d_model = 768, n_layers = 12, n_heads = 12
//   d_head = 64, d_ffn = 3072 (4 * d_model)
//
// Each transformer block:
//   x = x + Attention(LayerNorm(x))    [with KV-cache]
//   x = x + FFN(LayerNorm(x))
//
// GPT-2 uses Pre-LN (LayerNorm before each sub-layer)
// ---------------------------------------------------------------------------

// Model hyperparameters — hardcoded for GPT-2 small
// TODO: load from config.json for other model sizes
struct GPT2Config {
    int vocab_size  = 50257;
    int d_model     = 768;
    int n_layers    = 12;
    int n_heads     = 12;
    int d_head      = 64;    // d_model / n_heads
    int d_ffn       = 3072;  // 4 * d_model
    int max_seq_len = 1024;
    int eos_token   = 50256; // <|endoftext|>
};

struct GPT2Model::Impl {
    GPT2Config   cfg;
    Precision    precision;
    int          max_seq_len;

    // Device weight buffers — allocated in load_weights()
    // Naming follows HuggingFace GPT-2 weight keys
    float* d_wte   = nullptr;   // token embedding  [vocab x d_model]
    float* d_wpe   = nullptr;   // pos embedding    [max_seq x d_model]
    float* d_ln_f_g = nullptr;  // final layernorm gamma [d_model]
    float* d_ln_f_b = nullptr;  // final layernorm beta  [d_model]

    // Per-layer weights — indexed by layer
    struct LayerWeights {
        float* ln1_g = nullptr;   // pre-attn layernorm gamma  [d_model]
        float* ln1_b = nullptr;   // pre-attn layernorm beta   [d_model]
        float* qkv_w = nullptr;   // QKV projection weight     [3*d_model x d_model]
        float* qkv_b = nullptr;   // QKV projection bias       [3*d_model]
        float* proj_w = nullptr;  // output projection weight  [d_model x d_model]
        float* proj_b = nullptr;  // output projection bias    [d_model]
        float* ln2_g = nullptr;   // pre-ffn layernorm gamma   [d_model]
        float* ln2_b = nullptr;   // pre-ffn layernorm beta    [d_model]
        float* fc1_w = nullptr;   // FFN up-projection         [d_ffn x d_model]
        float* fc1_b = nullptr;   // FFN up-projection bias    [d_ffn]
        float* fc2_w = nullptr;   // FFN down-projection       [d_model x d_ffn]
        float* fc2_b = nullptr;   // FFN down-projection bias  [d_model]
    };
    std::vector<LayerWeights> layers;

    // Activation buffers — reused across layers, sized for max_seq_len
    float* d_hidden  = nullptr;  // [max_seq x d_model]
    float* d_normed  = nullptr;  // [max_seq x d_model]
    float* d_qkv     = nullptr;  // [max_seq x 3*d_model]
    float* d_attn_out = nullptr; // [max_seq x d_model]
    float* d_ffn_mid = nullptr;  // [max_seq x d_ffn]
    float* d_logits  = nullptr;  // [vocab_size]

    // KV cache storage (Q, K, V per head after projection)
    float* d_q_buf   = nullptr;  // [n_heads x seq x d_head]
    float* d_k_buf   = nullptr;
    float* d_v_buf   = nullptr;
    float* d_o_buf   = nullptr;

    Impl(const std::string& model_path, Precision prec, int max_sl)
        : precision(prec), max_seq_len(max_sl)
    {
        layers.resize(cfg.n_layers);
        load_weights(model_path);
        allocate_activations();
    }

    ~Impl() { free_all(); }

    void load_weights(const std::string& path) {
        // TODO: implement weight loading from .bin file (HuggingFace safetensors
        // or raw binary dump from convert_weights.py)
        // For now: allocate and zero-initialize as placeholder
        // A real implementation reads: wte, wpe, and per-layer ln/qkv/proj/ffn weights

        auto alloc = [](float** ptr, size_t n) {
            cudaMalloc(ptr, n * sizeof(float));
            cudaMemset(*ptr, 0, n * sizeof(float));
        };

        alloc(&d_wte,    (size_t)cfg.vocab_size  * cfg.d_model);
        alloc(&d_wpe,    (size_t)cfg.max_seq_len * cfg.d_model);
        alloc(&d_ln_f_g, cfg.d_model);
        alloc(&d_ln_f_b, cfg.d_model);

        for (int l = 0; l < cfg.n_layers; l++) {
            auto& lw = layers[l];
            alloc(&lw.ln1_g,  cfg.d_model);
            alloc(&lw.ln1_b,  cfg.d_model);
            alloc(&lw.qkv_w,  3 * cfg.d_model * cfg.d_model);
            alloc(&lw.qkv_b,  3 * cfg.d_model);
            alloc(&lw.proj_w, cfg.d_model * cfg.d_model);
            alloc(&lw.proj_b, cfg.d_model);
            alloc(&lw.ln2_g,  cfg.d_model);
            alloc(&lw.ln2_b,  cfg.d_model);
            alloc(&lw.fc1_w,  cfg.d_ffn * cfg.d_model);
            alloc(&lw.fc1_b,  cfg.d_ffn);
            alloc(&lw.fc2_w,  cfg.d_model * cfg.d_ffn);
            alloc(&lw.fc2_b,  cfg.d_model);
        }
    }

    void allocate_activations() {
        int S = max_seq_len;
        int D = cfg.d_model;
        int F = cfg.d_ffn;
        int V = cfg.vocab_size;
        int H = cfg.n_heads;
        int dh = cfg.d_head;

        auto alloc = [](float** ptr, size_t n) {
            cudaMalloc(ptr, n * sizeof(float));
        };

        alloc(&d_hidden,   (size_t)S * D);
        alloc(&d_normed,   (size_t)S * D);
        alloc(&d_qkv,      (size_t)S * 3 * D);
        alloc(&d_attn_out, (size_t)S * D);
        alloc(&d_ffn_mid,  (size_t)S * F);
        alloc(&d_logits,   V);
        alloc(&d_q_buf,    (size_t)H * S * dh);
        alloc(&d_k_buf,    (size_t)H * S * dh);
        alloc(&d_v_buf,    (size_t)H * S * dh);
        alloc(&d_o_buf,    (size_t)H * S * dh);
    }

    void free_all() {
        auto f = [](float* p) { if (p) cudaFree(p); };
        f(d_wte); f(d_wpe); f(d_ln_f_g); f(d_ln_f_b);
        for (auto& lw : layers) {
            f(lw.ln1_g); f(lw.ln1_b); f(lw.qkv_w); f(lw.qkv_b);
            f(lw.proj_w); f(lw.proj_b); f(lw.ln2_g); f(lw.ln2_b);
            f(lw.fc1_w); f(lw.fc1_b); f(lw.fc2_w); f(lw.fc2_b);
        }
        f(d_hidden); f(d_normed); f(d_qkv); f(d_attn_out);
        f(d_ffn_mid); f(d_logits);
        f(d_q_buf); f(d_k_buf); f(d_v_buf); f(d_o_buf);
    }
};


// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
std::vector<float> GPT2Model::forward(
    const std::vector<int>& input_ids,
    PagedKVCache& cache)
{
    int S  = (int)input_ids.size();
    int D  = impl_->cfg.d_model;
    int H  = impl_->cfg.n_heads;
    int dh = impl_->cfg.d_head;
    int F  = impl_->cfg.d_ffn;
    int V  = impl_->cfg.vocab_size;

    // Copy token ids to device
    int* d_ids = nullptr;
    cudaMalloc(&d_ids, S * sizeof(int));
    cudaMemcpy(d_ids, input_ids.data(), S * sizeof(int), cudaMemcpyHostToDevice);

    // --- Embedding: token + positional ---
    int pos_offset = cache.current_seq_len();
    launch_token_embedding(d_ids, impl_->d_wte, impl_->d_hidden,
                           S, impl_->cfg.vocab_size, D);
    launch_positional_embedding(impl_->d_wpe, impl_->d_hidden,
                                S, D, pos_offset);
    cudaFree(d_ids);

    // --- Transformer layers ---
    for (int l = 0; l < impl_->cfg.n_layers; l++) {
        auto& lw = impl_->layers[l];

        // Pre-attention LayerNorm + QKV projection (fused)
        launch_layernorm_linear_fused(
            impl_->d_hidden, lw.ln1_g, lw.ln1_b,
            lw.qkv_w, lw.qkv_b, impl_->d_qkv,
            S, D, 3 * D);

        // Split QKV into separate heads for attention
        // d_qkv: [S x 3D] → q,k,v each [H x S x dh]
        // (reshape + transpose — simplified here as pointer offsets)
        float* d_q = impl_->d_q_buf;
        float* d_k = impl_->d_k_buf;
        float* d_v = impl_->d_v_buf;

        // TODO: proper head-split transpose kernel
        // For now: treat as [1 x H x S x dh] for flash attention
        launch_flash_attention(d_q, d_k, d_v, impl_->d_o_buf,
                               1, H, S, dh, /*causal=*/true);

        // Output projection: [S x D] → [S x D]
        launch_linear(impl_->d_o_buf, lw.proj_w, lw.proj_b,
                      impl_->d_attn_out, S, D, D);

        // Residual connection: hidden += attn_out
        launch_add(impl_->d_hidden, impl_->d_attn_out, S * D);

        // Pre-FFN LayerNorm + FC1 (fused)
        launch_layernorm_linear_fused(
            impl_->d_hidden, lw.ln2_g, lw.ln2_b,
            lw.fc1_w, lw.fc1_b, impl_->d_ffn_mid,
            S, D, F);

        // GELU activation in-place (tanh approximation — matches GPT-2 training)
        launch_gelu(impl_->d_ffn_mid, S * F);

        // FC2: [S x F] → [S x D]
        launch_linear(impl_->d_ffn_mid, lw.fc2_w, lw.fc2_b,
                      impl_->d_attn_out, S, F, D);

        // Residual: hidden += ffn_out
        launch_add(impl_->d_hidden, impl_->d_attn_out, S * D);
    }

    // --- Final LayerNorm ---
    launch_layernorm(impl_->d_hidden, impl_->d_ln_f_g, impl_->d_ln_f_b,
                     impl_->d_normed, S, D);

    // --- LM head: last token hidden → logits [vocab_size] ---
    // GPT-2 ties lm_head to wte (weight sharing)
    float* d_last = impl_->d_normed + (S - 1) * D;
    launch_linear(d_last, impl_->d_wte, /*bias=*/nullptr,
                  impl_->d_logits, 1, D, V);

    // Copy logits to host
    std::vector<float> logits(V);
    cudaMemcpy(logits.data(), impl_->d_logits,
               V * sizeof(float), cudaMemcpyDeviceToHost);

    return logits;
}


// ---------------------------------------------------------------------------
// Tokenizer stubs — replace with real BPE tokenizer
// ---------------------------------------------------------------------------
std::vector<int> GPT2Model::tokenize(const std::string& text) const {
    // TODO: load vocab.json + merges.txt, implement BPE
    // Placeholder: return ASCII codes (incorrect, for structure only)
    std::vector<int> ids;
    for (char c : text) ids.push_back((int)(unsigned char)c);
    return ids;
}

std::string GPT2Model::detokenize(const std::vector<int>& ids) const {
    // TODO: reverse BPE lookup
    std::string out;
    for (int id : ids) out += (char)id;
    return out;
}


// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
GPT2Model::GPT2Model(const std::string& path, Precision prec, int max_seq)
    : impl_(new Impl(path, prec, max_seq)) {}

GPT2Model::~GPT2Model() { delete impl_; }

int GPT2Model::eos_token_id() const { return impl_->cfg.eos_token; }
int GPT2Model::vocab_size()   const { return impl_->cfg.vocab_size; }
int GPT2Model::num_layers()   const { return impl_->cfg.n_layers; }
int GPT2Model::num_heads()    const { return impl_->cfg.n_heads; }
int GPT2Model::head_dim()     const { return impl_->cfg.d_head; }
int GPT2Model::d_model()      const { return impl_->cfg.d_model; }