#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "gguf.hpp"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>
#include <vector>
#include <optional>

using json = nlohmann::json;

struct ModelData;
struct ModelParams;
struct TokenizerParams;
struct RuntimeParams;

enum class DataType { F32 = 0, F16 = 1 };
enum class Device { CPU, CUDA };

struct Tensor {
    mutable void* data = nullptr;
    std::string name;
    std::array<uint64_t,4> shape = {0,0,0,0};
    DataType dtype;
    uint64_t size = 0;
    mutable Device device = Device::CPU;
    
    void to(Device target_device);
    json to_json() const;
};

struct ModelData {
    json metadata;
    json tensor_metadata;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    int from_file(const std::string& filename);
};

namespace ModelSupport {
    inline const std::unordered_set<std::string> SUPPORTED_ARCHITECTURES = {"qwen3"};
}

struct RuntimeParams {
    uint64_t max_seq_len, top_k, seed;
    float temperature, top_p;
    
    RuntimeParams(uint64_t max_seq_len, float temperature, uint64_t top_k, float top_p, uint64_t seed)
        : max_seq_len(max_seq_len), top_k(top_k), seed(seed), temperature(temperature), top_p(top_p) {}
};

struct Config {
    // model params
    uint64_t vocab_size, model_max_seq_len, embed_dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim;
    float rms_norm_eps, theta;
    uint32_t moe_top_k, n_experts, expert_dim;
    bool is_moe;

    // tokenizer params
    std::string tokenizer_model, tokenizer_pre, chat_template;
    std::vector<std::string> tokens, merges;
    std::vector<uint32_t> token_type;
    uint32_t eos_token_id, padding_token_id;
    bool add_bos_token;
    
    // run params
    uint32_t user_max_seq_len, top_k, seed;
    float temperature, top_p;
    
    Config(const ModelData& model_data, const RuntimeParams& runtime_params);
};

struct Pool {
    Device device;
    
    float *x, *xb, *xb2;                           // activations
    float *hb, *hb2;                               // ffn
    float *q, *k, *v, *att;                        // attention
    float *moe_weights, *active_experts_weights;   // MoE (TODO)
    int *active_experts;                           // MoE indices (TODO)
    float *logits;                                 // output
    
    Pool(const Config& config);
    ~Pool();
    void to(Device device);
};

uint64_t dtype_size(DataType dtype);
std::string dtype_to_str(DataType dtype);
std::optional<DataType> tensor_to_data_type(TensorType t_type);

#endif