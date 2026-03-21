#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <random>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Engine headers (C++ interface)
#include "../engine/model.h"
#include "../engine/kvcache.h"
#include "../engine/sampler.h"

// Kernel lauch declarations (defined in .cu files, linked by CMake)
#include "../kernels/gemm.h"
#include "../kernels/softmax.h"
#include "../kernels/attention.h"
#include "../kernels/quantize.h"

namespace py = pybind11;
using namespace std;

// ----------------------------------------------------------------------
// Helper: numpy array (CPU) <-> CUDA device point round-trip
// ----------------------------------------------------------------------

// Copy a float32 numpy array to a freshly allocated CUDA buffer
// Caller owns the returned point and must cudaFree it
float* numpy_to_device(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    size_t n_bytes = buf.size * sizeof(float);

    float* d_ptr = nullptr;
    cudaMalloc(&d_ptr, n_bytes);
    cudaMemcpy(d_ptr, buf.ptr, n_bytes, cudaMemcpyHostToDevice);
    return d_ptr;
}

// Copy a CUDA buffer back to a new numpy array
py::array_t<float> device_to_numpy(const float* d_ptr, vector<ssize_t> shape) {
    size_t n = 1;
    for (auto s : shape) n *= s;

    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// ---------------------------------------------------------------
// Thin Python wrappers arounnd individual CUDA kernels 
// ---------------------------------------------------------------

// Expose as nanoinfer.kernels.gemm(A, B) -> C
py::array_t<float> py_gemm(py::array_t<float> A, py::array_t<float> B) {
    auto a_buf = A.request(), b_buf = B.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2)
        throw runtime_error("gemm: inputs must be 2-D arrays");

    int M = a_buf.shape[0];
    int K = a_buf.shape[1];
    int N = b_buf.shape[1];

    if (b_buf.shape[0] != K)
        throw runtime_error("gemm: inner dimensions do not match");

    float* d_A = numpy_to_device(A);
    float* d_B = numpy_to_device(B);
    float* d_C = nullptr;
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemset(d_C, 0, M * N * sizeof(float));

    launch_gemm_tiled(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    auto result = device_to_numpy(d_C, {M, N});

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return result;
}

// Exposed as nanoinfer.kernels.softmax(x) -> y
// Applies row-wise softmax using the online (single-pass) kernel.
py::array_t<float> py_softmax(py::array_t<float> x) {
    auto buf = x.request();
    if (buf.ndim != 2)
        throw runtime_error("softmax: input must be 2-D (rows x cols)");

    int rows = buf.shape[0];
    int cols = buf.shape[1];

    float* d_x = numpy_to_device(x);
    float* d_y = nullptr;
    cudaMalloc(&d_y, rows * cols * sizeof(float));

    launch_softmax_online(d_x, d_y, rows, cols);
    cudaDeviceSynchronize();

    auto result = device_to_numpy(d_y, {rows, cols});

    cudaFree(d_x);
    cudaFree(d_y);
    return result;
}

// Exposed as nanoinfer.kernels.flash_attention(Q, K, V, causal) -> 0
py::array_t<float> py_flash_attention(
    py::array_t<float> Q,
    py::array_t<float> K,
    py::array_t<float> V,
    bool causal = true
) {
    auto q_buf = Q.request();
    // Expected shape: [batch, heads, seq_len, d_head]
    if (q_buf.ndim != 4)
        throw runtime_error("flash_attention: Q/K/V must be 4-D [B, H, S, d]");

    int B  = q_buf.shape[0];
    int H  = q_buf.shape[1];
    int S  = q_buf.shape[2];
    int d  = q_buf.shape[3];
    int n  = B * H * S * d;
 
    float* d_Q = numpy_to_device(Q);
    float* d_K = numpy_to_device(K);
    float* d_V = numpy_to_device(V);
    float* d_O = nullptr;
    cudaMalloc(&d_O, n * sizeof(float));

    launch_flash_attention(d_Q, d_K, d_V, d_O, B, H, S, d, causal);
    cudaDeviceSynchronize();

    auto result = device_to_numpy(d_O, {B, H, S, d});

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    return result;
}

// -----------------------------------------------------------------
// NanoInfer — top-level inference engine class
// -----------------------------------------------------------------

// Wraps engine/model.h: GPT2Model
// Exposed as nanoinfer.NanoInfer
class PyNanoInfer {
public:
    explicit PyNanoInfer(const std::string& model_path,
                         const std::string& precision,
                         int    max_seq_len,
                         int    kvcache_pages)
        : precision_(precision)
    {
        Precision prec = Precision::FP32;
        if (precision == "fp16") prec = Precision::FP16;
        else if (precision == "int8") prec = Precision::INT8;
        else if (precision != "fp32")
            throw runtime_error("precision must be 'fp32', 'fp16', or 'int8'");
        
        model_ = make_unique<GPT2Model>(model_path, prec, max_seq_len);
        cache_ = make_unique<PagedKVCache>(
            model_->num_layers(),
            model_->num_heads(),
            model_->head_dim(),
            kvcache_pages
        );
    }

    //nanoinfer.NanoInfer.from_pretrained("gpt2", precision="fp16")
    static shared_ptr<PyNanoInfer> from_pretrained(
        const std::string& model_name,
        const std::string& precision = "fp32",
        int max_seq_len = 2048,
        int kvcache_pages = 256
    ) {
        // resolve model_name -> local weights path
        // for now: treat model_name as a path directly
        // TODO: add HuggingFace Hub download helper
        return make_shared<PyNanoInfer>(
            model_name, precision, max_seq_len, kvcache_pages);
    }

    // nanoinfer.NanoInfer.generate(prompt, max_tokens, temperature, top_p)
    string generate(
        const std::string& prompt,
        int max_tokens = 100,
        float temperature = 1.0f,
        float top_p = 0.95f,
        int top_k = 50,
        int seed = -1
    ) {
        // tokenize (simple BPE stub - replace with real tokenizer)
        vector<int> input_ids = model_->tokenize(prompt);

        SamplerConfig cfg;
        cfg.temperature = temperature;
        cfg.top_p = top_p;
        cfg.top_k = top_k;
        cfg.seed = (seed < 0) ? random_device{}() : (uint32_t)seed;

        Sampler sampler(cfg);
        cache_->reset();

        vector<int> output_ids;
        output_ids.reserve(max_tokens);

        for (int step = 0; step < max_tokens; ++step) {
            // forward pass: returns logits [vocab_size]
            vector<float> logits = model_->forward(input_ids, *cache_);

            int next_token = sampler.sample(logits);
            if (next_token == model_->eos_token_id()) break;
 
            output_ids.push_back(next_token);
            input_ids = {next_token};  // subsequent steps: single new token
        }

        return model_->detokenize(output_ids);
    }

    // nanoinfer.NanoInfer.encode(text) -> token ids as list
    vector<int> encode(const string& text) {
        return model_->tokenize(text);
    }

    // nanoinfer.NanoInfer.decode(ids) -> text
    std::string decode(const std::vector<int>& ids) {
        return model_->detokenize(ids);
    }

    // properties
    std::string precision() const { return precision_; }
    int vocab_size() const { return model_->vocab_size(); }
    int num_layers() const { return model_->num_layers(); }
    int num_heads() const { return model_->num_heads(); }
    int d_model() const { return model_->d_model(); }

private:
    string precision_;
    unique_ptr<GPT2Model> model_;
    unique_ptr<PagedKVCache> cache_;
};

// -------------------------------------------------------
// pybind11 module definition
// -------------------------------------------------------

PYBIND11_MODULE(nanoinfer, m) {
    m.doc() = "NanoInfer: lightweight CUDA inference engine for transformers";
 
    // -- sub-module: low-level kernel access (for benchmarking & testing) --
    auto kernels = m.def_submodule("kernels", "Raw CUDA kernel wrappers");

    kernels.def("gemm", &py_gemm,
        py::arg("A"), py::arg("B"),
        R"doc(
            Tiled GEMM: C = A @ B.
            A: (M, K) float32 numpy array
            B: (K, N) float32 numpy array
            Returns C: (M, N) float32 numpy array
        )doc");
    
    kernels.def("softmax", &py_softmax,
        py::arg("x"),
        R"doc(
            Row-wise online softmax (single-pass, numerically stable).
            x: (rows, cols) float32 numpy array
            Returns y: same shape, each row sums to 1.
        )doc");

    kernels.def("flash_attention", &py_flash_attention,
        py::arg("Q"), py::arg("K"), py::arg("V"),
        py::arg("causal") = true,
        R"doc(
            Flash Attention v1: IO-aware exact attention.
            Q, K, V: (batch, heads, seq_len, d_head) float32 numpy arrays
            causal: apply causal mask (default True for autoregressive use)
            Returns O: same shape as Q
        )doc");

    // -- top-level NanoInfer class --
    py::class_<PyNanoInfer, shared_ptr<PyNanoInfer>>(m, "NanoInfer")
        .def(py::init<const std::string&, const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("precision")     = "fp32",
             py::arg("max_seq_len")   = 2048,
             py::arg("kvcache_pages") = 256)

        .def_static("from_pretrained", &PyNanoInfer::from_pretrained,
             py::arg("model_name"),
             py::arg("precision")     = "fp32",
             py::arg("max_seq_len")   = 2048,
             py::arg("kvcache_pages") = 256,
             R"doc(
                 Load a pre-trained model by name or local path.
 
                 Example:
                     model = NanoInfer.from_pretrained("gpt2", precision="int8")
             )doc")

        .def("generate", &PyNanoInfer::generate,
             py::arg("prompt"),
             py::arg("max_tokens")  = 100,
             py::arg("temperature") = 1.0f,
             py::arg("top_p")       = 0.95f,
             py::arg("top_k")       = 50,
             py::arg("seed")        = -1,
             R"doc(
                 Generate text continuation from a prompt.
 
                 Args:
                     prompt:      Input string.
                     max_tokens:  Maximum new tokens to generate.
                     temperature: Sampling temperature (0 = greedy, 1 = normal).
                     top_p:       Nucleus sampling threshold.
                     top_k:       Top-k cutoff (0 = disabled).
                     seed:        RNG seed for reproducibility (-1 = random).
 
                 Returns:
                     Generated text string (not including the prompt).
             )doc")

        .def("encode", &PyNanoInfer::encode,
             py::arg("text"),
             "Tokenize text -> list of integer token ids.")
 
        .def("decode", &PyNanoInfer::decode,
             py::arg("ids"),
             "Detokenize list of token ids -> string.")
 
        // read-only properties
        .def_property_readonly("precision", &PyNanoInfer::precision)
        .def_property_readonly("vocab_size", &PyNanoInfer::vocab_size)
        .def_property_readonly("num_layers", &PyNanoInfer::num_layers)
        .def_property_readonly("num_heads", &PyNanoInfer::num_heads)
        .def_property_readonly("d_model", &PyNanoInfer::d_model)
 
        .def("__repr__", [](const PyNanoInfer& m) {
            return "<NanoInfer precision=" + m.precision() +
                   " layers=" + std::to_string(m.num_layers()) +
                   " d=" + std::to_string(m.d_model()) + ">";
        });
}