#ifndef MODULE_HPP
#define MODULE_HPP
#include "config.hpp"
#include <unordered_map>
#include <any>

class Module {
protected:
    RootModule& root;
    void add_parameter(const Tensor& tensor); // wrapper just in case we need to debug
public:
    Module(RootModule& root);
    virtual ~Module() = default;
    virtual void forward(Pool& pool) = 0;
};

class RootModule {
private:
    std::vector<const Tensor*> parameters;
    Device device = Device::CPU;

public:
    virtual ~RootModule() = default;
    void add_parameter(const Tensor& tensor);
    Device get_device() const;
    void to(const Device device);
    virtual void forward() = 0;
};

// Base classes, pure virtual forward
class RoPE : public Module {
protected:
    uint64_t head_dim;
    uint64_t max_seq_len;
    float theta;
    
public:
    RoPE(RootModule& root, uint64_t head_dim, uint64_t max_seq_len, float theta);
    void forward(Pool& pool) override = 0;
};

class Linear : public Module {
protected:
    uint64_t in_features;
    uint64_t out_features;
    const Tensor& weight;
    const Tensor* bias;
    
public:
    Linear(RootModule& root, uint64_t in_features, uint64_t out_features,
           const Tensor& weight, const Tensor* bias = nullptr);
    void forward(Pool& pool) override = 0;
};

class RMSNorm : public Module {
protected:
    uint64_t dim;
    float eps;
    const Tensor& weight;
    
public:
    RMSNorm(RootModule& root, uint64_t dim, float eps, const Tensor& weight);
    void forward(Pool& pool) override = 0;
};

// Struct for easier parsing
struct DBTensors {
    const Tensor& wq;               // q_proj_weight  
    const Tensor& wk;               // k_proj_weight
    const Tensor& wv;               // v_proj_weight
    const Tensor& wo;               // o_proj_weight
    const Tensor& q_norm;           // q_norm_weight
    const Tensor& k_norm;           // k_norm_weight
    
    // FFN (nullptr if MoE)
    const Tensor* w1;               // gate_proj
    const Tensor* w2;               // down_proj
    const Tensor* w3;               // up_proj
    
    // MoE (nullptr if dense)
    const Tensor* gate;                    // Router weights
    const std::vector<Tensor>* expert_w1;  // Per-expert gate weights
    const std::vector<Tensor>* expert_w2;  // Per-expert down weights  
    const std::vector<Tensor>* expert_w3;  // Per-expert up weights
    
    // LN
    const Tensor& attn_norm;           // input_layernorm
    const Tensor& ffn_norm;            // post_attention_layernorm  
};

class DBBlock : public Module {
protected:
    uint64_t layer_idx;
    uint64_t n_heads;
    uint64_t n_kv_heads;
    uint64_t head_dim;
    uint64_t dim;
    uint64_t hidden_dim;
    float eps;
    uint64_t max_seq_len;
    float theta;
    // MoE
    uint64_t n_experts;
    uint64_t n_active_experts;
    bool is_moe_layer;
    // Tensors
    DBTensors tensors;
    
public:
    DBBlock(RootModule& root, uint64_t layer_idx, uint64_t n_heads, uint64_t n_kv_heads,
            uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
            float eps, uint64_t max_seq_len, float theta,
            uint64_t n_experts, uint64_t n_active_experts, bool is_moe_layer,
            DBTensors tensors);
    
    void forward(Pool& pool) override = 0;
};

// Qwen3-specific implementations

// Used for embedding, lm_head
class Qwen3Linear : public Linear {
public:
    Qwen3Linear(RootModule& root, uint64_t in_features, uint64_t out_features,
                const Tensor& weight, const Tensor* bias = nullptr);
    void forward(Pool& pool) override;
};

// Used after embedding, before first decoder block, inside decoder blocks
class Qwen3RoPE : public RoPE {
public:
    Qwen3RoPE(RootModule& root, uint64_t head_dim, uint64_t max_seq_len, float theta);
    void forward(Pool& pool) override;
};

// Decoder blocks in Qwen3, can support either dense or MoE
class Qwen3DB : public DBBlock {
public:
    Qwen3DB(RootModule& root, uint64_t layer_idx, uint64_t n_heads, uint64_t n_kv_heads,
            uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
            float eps, uint64_t max_seq_len, float theta,
            uint64_t n_experts, uint64_t n_active_experts, bool is_moe_layer,
            DBTensors tensors);
    
    void forward(Pool& pool) override;
};

// Used throughout the model
class Qwen3RMSNorm : public RMSNorm {
public:
    Qwen3RMSNorm(RootModule& root, uint64_t dim, float eps, const Tensor& weight);
    void forward(Pool& pool) override;
};

#endif