#pragma once

#include "minfer/parsing/gguf.hpp"
#include "extern/nlohmann/json_fwd.hpp"

#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <chrono>

// forward decls.
using minfer_json = nlohmann::json;
struct Tensor;
namespace MTL { class Buffer; }
enum class DataType : int;

enum class DeviceType { CPU = 0, METAL = 1 };

namespace ModelSupport {
    inline const std::unordered_set<std::string> SUPPORTED_ARCHITECTURES = {"qwen3"};
}

struct ModelData {
    std::unique_ptr<minfer_json> metadata;
    std::unique_ptr<minfer_json> tensor_metadata;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    GGUFFile gguf_file;
    int from_file(const std::string& filename);
    ~ModelData();
};

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

struct RunState {
    std::shared_ptr<Config> config;
    DeviceType device;
    int cur_pos;
    uint32_t token_id;
    bool compute_logits;

    std::variant<float*, MTL::Buffer*> x, xb, xb2;
    std::variant<float*, MTL::Buffer*> hb, hb2;
    std::variant<float*, MTL::Buffer*> q, k, v, att_scores, att_out;
    std::variant<float*, MTL::Buffer*> k_cache, v_cache;
    std::variant<float*, MTL::Buffer*> moe_scores;
    std::variant<int*, MTL::Buffer*> active_experts;
    std::variant<float*, MTL::Buffer*> active_experts_weights;
    std::variant<float*, MTL::Buffer*> logits;
    
    size_t kv_bytes_per_pos;

    explicit RunState(const std::shared_ptr<Config> config);
    ~RunState();

    void set_device(DeviceType target_device);

    void to_metal();
    void from_metal();
};

DataType tensor_to_data_type(TensorType t_type);
size_t dtype_size(DataType dtype);
std::string dtype_to_str(DataType dtype);
DataType str_to_dtype(std::string& dtype_str);
std::string device_to_str(DeviceType device);
std::string dtype_kernel_suffix(DataType dtype);