#ifndef MODULE_HPP
#define MODULE_HPP
#include "config.hpp"
#include <unordered_map>

class BaseLayer {
    friend class BaseModel; // for access and debug in models

private:
    std::vector<std::shared_ptr<Tensor>> parameters;
    Device device = Device::CPU;

protected:
    BaseLayer() = default;
    void append_parameter(std::shared_ptr<Tensor> tensor);
    virtual void forward(Pool& pool) = 0;
    void to(Device target_device);
    Device get_device() const;

public:
    virtual ~BaseLayer() = default; // need virtual public destructor
};

// Base classes, pure virtual forward
class Linear : public BaseLayer {
protected:
    uint64_t in_features;
    uint64_t out_features;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    Linear(uint64_t in_features, uint64_t out_features,
           std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr);
    virtual void forward(Pool& pool) override = 0;
};

class RoPE : public BaseLayer {
protected:
    uint64_t head_dim;
    uint64_t max_seq_len;
    float theta;
    RoPE(uint64_t head_dim, uint64_t max_seq_len, float theta);
    virtual void forward(Pool& pool) override = 0;
};

class RMSNorm : public BaseLayer {
protected:
    uint64_t dim;
    float eps;
    std::shared_ptr<Tensor> weight;
    RMSNorm(uint64_t dim, float eps, std::shared_ptr<Tensor> weight);
    virtual void forward(Pool& pool) override = 0;
};

// Struct for easier parsing
struct DBTensors {
    std::shared_ptr<Tensor> wq, wk, wv, wo, q_norm, k_norm;
    std::shared_ptr<Tensor> ffn_gate, ffn_down, ffn_up;
    std::shared_ptr<Tensor> moe_gate, expert_gate, expert_down, expert_up;
    std::shared_ptr<Tensor> attn_norm, ffn_norm;
};

class DBBlock : public BaseLayer {
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
    DBBlock(uint64_t block_idx, uint64_t n_heads, uint64_t n_kv_heads,
            uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
            float eps, uint64_t max_seq_len, float theta,
            uint64_t n_experts, uint64_t n_active_experts, bool is_moe_block,
            DBTensors tensors);
    
    virtual void forward(Pool& pool) override = 0;
};

// Qwen3-specific implementations

// Used for embedding, lm_head
class Qwen3Linear : public Linear {
public:
    Qwen3Linear(uint64_t in_features, uint64_t out_features,
                std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr);
    void forward(Pool& pool) override;
};

// Used after embedding, before first decoder block, inside decoder blocks
class Qwen3RoPE : public RoPE {
public:
    Qwen3RoPE(uint64_t head_dim, uint64_t max_seq_len, float theta);
    void forward(Pool& pool) override;
};

// Decoder blocks in Qwen3, can support either dense or MoE
class Qwen3DB : public DBBlock {
public:
    Qwen3DB(uint64_t block_idx, uint64_t n_heads, uint64_t n_kv_heads,
            uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
            float eps, uint64_t max_seq_len, float theta,
            uint64_t n_experts, uint64_t n_active_experts, bool is_moe_block,
            DBTensors tensors);
    
    void forward(Pool& pool) override;
};

// Used throughout the model
class Qwen3RMSNorm : public RMSNorm {
public:
    Qwen3RMSNorm(uint64_t dim, float eps, std::shared_ptr<Tensor> weight);
    void forward(Pool& pool) override;
};

#endif