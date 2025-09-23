#pragma once

#include "minfer/parsing/gguf.hpp"
#include "extern/nlohmann/json.hpp"

#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <chrono>

using json = nlohmann::ordered_json;

enum class DataType { F32 = 0, F16 = 1, BF16 = 2, INVALID = 3 };
enum class DeviceType { CPU, METAL };

struct Tensor {
    void* data = nullptr;
    std::string name;
    std::array<int,4> shape = {0,0,0,0};
    DataType dtype;
    size_t size_bytes = 0;
    DeviceType device = DeviceType::CPU;
    
    void set_device(DeviceType target_device);
    json to_json() const;

    void to_metal();
    void from_metal();
};

struct ModelData {
    json metadata;
    json tensor_metadata;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    GGUFFile gguf_file;
    int from_file(const std::string& filename);
};

namespace ModelSupport {
    inline const std::unordered_set<std::string> SUPPORTED_ARCHITECTURES = {"qwen3"};
}

struct RunParams {
    size_t max_seq_len, top_k, num_iters;
    int seed;
    float temperature, top_p, min_p, penalty_pres;
    
    RunParams(
        size_t num_iters,
        size_t max_seq_len,
        float temperature, 
        size_t top_k,
        float top_p,
        float min_p,
        float penalty_pres,
        int seed
    );
};

struct Config {

    // model params
    size_t vocab_size, model_max_seq_len;
    int d_model, n_layers,
        n_heads, n_kv_heads, d_head, d_rotary,
        d_ff, n_active_experts, n_experts;
    float rms_norm_eps, freq_base;

    // tokenizer params
    std::string tokenizer_model, tokenizer_pre, chat_template;
    std::vector<std::string> tokens, merges;
    std::vector<uint32_t> token_type;
    uint32_t eos_token_id, padding_token_id;
    bool add_bos_token;
    
    // run params
    size_t user_max_seq_len, top_k, num_iters;
    int seed;
    float temperature, top_p, min_p, penalty_pres;

    Config(const ModelData& model_data, const RunParams& runtime_params);
};

struct AlignedDeleter {
    void operator()(void* ptr) {
        std::free(ptr);
    }
};

struct RunState {
    std::shared_ptr<Config> config;
    DeviceType device;
    int cur_pos;            // cur pos in seq
    uint32_t token_id;      // token ID at current pos. in seq.
    bool compute_logits;    // prefill optimization

    std::unique_ptr<float[], AlignedDeleter> x, xb, xb2;                               // activations
    std::unique_ptr<float[], AlignedDeleter> hb, hb2;                                  // ffn
    std::unique_ptr<float[], AlignedDeleter> q, k, v, att_scores, att_out;             // attention
    std::unique_ptr<float[], AlignedDeleter> k_cache, v_cache;                         // k,v caches
    std::unique_ptr<float[], AlignedDeleter> moe_scores;                               // router scores for all experts
    std::unique_ptr<int[], AlignedDeleter> active_experts;                             // indices of topK (active) experts
    std::unique_ptr<float[], AlignedDeleter> active_experts_scores;                    // scores for topK (active) experts
    std::unique_ptr<float[], AlignedDeleter> active_experts_weights;                   // weights for topK (active) experts
    std::unique_ptr<float[], AlignedDeleter> logits;                                   // output logits at end
    
    // kv cache bytes per position
    size_t kv_bytes_per_pos;

    explicit RunState(const std::shared_ptr<Config> config);
    void set_device(DeviceType target_device);

    void to_metal();
    void from_metal();
};

struct GenStats {
    int num_tokens_gen = 0; // number of tokens generated
    float ttft = 0.0f;         // time to first token, sec
    float throughput = 0.0f;   // tok/sec
    float prefill_time = 0.0f; // in secs
    float bandwidth = 0.0f;    // in GB/sec

    std::chrono::high_resolution_clock::time_point timer_start;

    void start_timer();
    float get_elapsed_sec() const;
    void print_stats() const;
};

DataType tensor_to_data_type(TensorType t_type);
size_t dtype_size(DataType dtype);
std::string dtype_to_str(DataType dtype);
DataType str_to_dtype(std::string& dtype_str);
std::string device_to_str(DeviceType device);