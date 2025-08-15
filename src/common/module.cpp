#include "module.hpp"

// === BASE LAYER ===

void BaseLayer::append_parameter(std::shared_ptr<Tensor> tensor) {
    parameters.push_back(tensor);
}

void BaseLayer::to(Device target_device) {
    if (this->device == target_device) return;
    for (auto& param : parameters) {
        if (param) {
            // Move parameter tensor to target device
            param->to(target_device); // TO-DO, add support for this
        }
    }
    this->device = target_device;
}

Device BaseLayer::get_device() const {
    return this->device;
}

// === LAYER ===

Linear::Linear(uint64_t in_features, uint64_t out_features,
               std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias)
    : in_features(in_features), out_features(out_features), weight(weight), bias(bias) {
    append_parameter(weight);
    if (bias) {
        append_parameter(bias);
    }
}

RoPE::RoPE(uint64_t head_dim, uint64_t max_seq_len, float theta)
    : head_dim(head_dim), max_seq_len(max_seq_len), theta(theta) {}

RMSNorm::RMSNorm(uint64_t dim, float eps, std::shared_ptr<Tensor> weight)
    : dim(dim), eps(eps), weight(weight) {
    append_parameter(weight);
}

DBBlock::DBBlock(uint64_t layer_idx, uint64_t n_heads, uint64_t n_kv_heads,
                 uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
                 float eps, uint64_t max_seq_len, float theta,
                 uint64_t n_experts, uint64_t n_active_experts, bool is_moe_layer,
                 DBTensors tensors)
    : layer_idx(layer_idx), n_heads(n_heads), n_kv_heads(n_kv_heads),
      head_dim(head_dim), dim(dim), hidden_dim(hidden_dim), eps(eps),
      max_seq_len(max_seq_len), theta(theta), n_experts(n_experts),
      n_active_experts(n_active_experts), is_moe_layer(is_moe_layer),
      tensors(tensors) {
    
    append_parameter(tensors.wq);
    append_parameter(tensors.wk);
    append_parameter(tensors.wv);
    append_parameter(tensors.wo);
    append_parameter(tensors.q_norm);
    append_parameter(tensors.k_norm);
    // depending on if dense or MoE model
    if (tensors.ffn_gate) append_parameter(tensors.ffn_gate);
    if (tensors.ffn_down) append_parameter(tensors.ffn_down);
    if (tensors.ffn_up) append_parameter(tensors.ffn_up);
    if (tensors.moe_gate) append_parameter(tensors.moe_gate);
    if (tensors.expert_gate) append_parameter(tensors.expert_gate);
    if (tensors.expert_down) append_parameter(tensors.expert_down);
    if (tensors.expert_up) append_parameter(tensors.expert_up);
    append_parameter(tensors.attn_norm);
    append_parameter(tensors.ffn_norm);
}

// === MODEL-SPECIFIC LAYERS ===

Qwen3Linear::Qwen3Linear(uint64_t in_features, uint64_t out_features,
                         std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias)
    : Linear(in_features, out_features, weight, bias) {}

void Qwen3Linear::forward(Pool& pool) {
    // TODO: implement
}

Qwen3RoPE::Qwen3RoPE(uint64_t head_dim, uint64_t max_seq_len, float theta)
    : RoPE(head_dim, max_seq_len, theta) {}

void Qwen3RoPE::forward(Pool& pool) {
    // TODO: implement
}

Qwen3DB::Qwen3DB(uint64_t layer_idx, uint64_t n_heads, uint64_t n_kv_heads,
                 uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
                 float eps, uint64_t max_seq_len, float theta,
                 uint64_t n_experts, uint64_t n_active_experts, bool is_moe_layer,
                 DBTensors tensors)
    : DBBlock(layer_idx, n_heads, n_kv_heads, head_dim, dim, hidden_dim,
              eps, max_seq_len, theta, n_experts, n_active_experts, is_moe_layer, tensors) {}

void Qwen3DB::forward(Pool& pool) {
    // TODO: implement
}

Qwen3RMSNorm::Qwen3RMSNorm(uint64_t dim, float eps, std::shared_ptr<Tensor> weight)
    : RMSNorm(dim, eps, weight) {}

void Qwen3RMSNorm::forward(Pool& pool) {
    // TODO: implement
}