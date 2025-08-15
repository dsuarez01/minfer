#include "module.hpp"

////////////////// Module //////////////////
Module::Module(RootModule& root) : root(root) {}

void Module::add_parameter(const Tensor& tensor) {
    root.add_parameter(tensor);
}

////////////////// RootModule //////////////////
void RootModule::add_parameter(const Tensor& tensor) {

    parameters.push_back(&tensor);
}

void RootModule::to(const Device device) {
    this->device = device;
    for (auto& tensor : parameters) {
        tensor->device = device;
    }
}

Device RootModule::get_device() const {
    return device;
}

////////////////// Base Classes //////////////////
RoPE::RoPE(RootModule& root, uint64_t head_dim, uint64_t max_seq_len, float theta): 
    Module(root), head_dim(head_dim), max_seq_len(max_seq_len), theta(theta) {}

Linear::Linear(RootModule& root, uint64_t in_features, uint64_t out_features, 
               const Tensor& weight, const Tensor* bias)
    : Module(root), in_features(in_features), out_features(out_features), weight(weight), bias(bias) {
    root.add_parameter(weight);
    if (bias) {
        root.add_parameter(*bias);
    }
}

RMSNorm::RMSNorm(RootModule& root, uint64_t dim, float eps, const Tensor& weight)
    : Module(root), dim(dim), eps(eps), weight(weight) {
    root.add_parameter(weight);
}

DBBlock::DBBlock(RootModule& root, uint64_t layer_idx, uint64_t n_heads, uint64_t n_kv_heads,
                 uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
                 float eps, uint64_t max_seq_len, float theta,
                 uint64_t n_experts, uint64_t n_active_experts, bool is_moe_layer,
                 DBTensors tensors)
    : Module(root), layer_idx(layer_idx), n_heads(n_heads), n_kv_heads(n_kv_heads), 
      head_dim(head_dim), dim(dim), hidden_dim(hidden_dim),
      eps(eps), max_seq_len(max_seq_len), theta(theta),
      n_experts(n_experts), n_active_experts(n_active_experts), is_moe_layer(is_moe_layer),
      tensors(tensors) {
    
    add_parameter(tensors.wq);
    add_parameter(tensors.wk);
    add_parameter(tensors.wv);
    add_parameter(tensors.wo);
    add_parameter(tensors.q_norm);
    add_parameter(tensors.k_norm);
    add_parameter(tensors.attn_norm);
    add_parameter(tensors.ffn_norm);
    
    if (!is_moe_layer && tensors.ffn_gate && tensors.ffn_down && tensors.ffn_up) {
        add_parameter(*tensors.ffn_gate);
        add_parameter(*tensors.ffn_down);
        add_parameter(*tensors.ffn_up);
    }

    if (is_moe_layer && tensors.moe_gate) {
        add_parameter(*tensors.moe_gate);
        if (tensors.expert_gate) add_parameter(*tensors.expert_gate);
        if (tensors.expert_down) add_parameter(*tensors.expert_down);  
        if (tensors.expert_up) add_parameter(*tensors.expert_up);
    }
}

////////////////// Qwen3-Specific Impl. //////////////////
Qwen3Linear::Qwen3Linear(RootModule& root, uint64_t in_features, uint64_t out_features,
                         const Tensor& weight, const Tensor* bias)
    : Linear(root, in_features, out_features, weight, bias) {}

void Qwen3Linear::forward(Pool& pool) {
    // TODO (NOTE: check device)
    // e.g. matmul(input, weight.data, bias ? bias->data : nullptr, output, in_features, out_features);
}

Qwen3RoPE::Qwen3RoPE(RootModule& root, uint64_t head_dim, uint64_t max_seq_len, float theta)
    : RoPE(root, head_dim, max_seq_len, theta) {}

void Qwen3RoPE::forward(Pool& pool) {
    // TODO (NOTE: check device)
    // e.g. rope(pool.q, pool.k, head_dim, max_seq_len, theta, position);
}

Qwen3DB::Qwen3DB(RootModule& root, uint64_t layer_idx, uint64_t n_heads, uint64_t n_kv_heads,
                 uint64_t head_dim, uint64_t dim, uint64_t hidden_dim,
                 float eps, uint64_t max_seq_len, float theta,
                 uint64_t n_experts, uint64_t n_active_experts, bool is_moe_layer,
                 DBTensors tensors)
    : DBBlock(root, layer_idx, n_heads, n_kv_heads, head_dim, dim, hidden_dim,
              eps, max_seq_len, theta, n_experts, n_active_experts, is_moe_layer, tensors) {}

void Qwen3DB::forward(Pool& pool) {
    // TODO (NOTE: check device)
    // Call kernels like e.g.:
    // if (is_moe_layer) {
    //     fused_rmsnorm_moe_residual(pool.x, tensors.ffn_norm, tensors.gate, tensors.expert_w1, ...);
    // } else {
    //     fused_rmsnorm_swiglu_residual(pool.x, tensors.ffn_norm, tensors.w1, tensors.w2, tensors.w3, ...);
    // }
    
    // Will probably just define a whole kernel
    // (Encapsulate all perf details in cpu/cuda opn. files)
}

Qwen3RMSNorm::Qwen3RMSNorm(RootModule& root, uint64_t dim, float eps, const Tensor& weight)
    : RMSNorm(root, dim, eps, weight) {}

void Qwen3RMSNorm::forward(Pool& pool) {
    // TODO (NOTE: check device)
    // e.g. rmsnorm(pool.x, weight.data, pool.x, dim, eps);
}