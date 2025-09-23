#include "minfer/base/module.hpp"

// === BASE LAYER ===

void BaseLayer::append_parameter(TPtr tensor) {
    _parameters.push_back(tensor);
}

void BaseLayer::set_device(DeviceType target_device) {
    if (_device == target_device) return;
    for (auto& param : _parameters) {
        if (param) {
            param->set_device(target_device);
        }
    }
    _device = target_device;
}

// === EMBED ===

Embed::Embed(
    size_t vocab_size, int d_model, 
    TPtr weight, 
    DataType qdtype, DeviceType device
) : vocab_size(vocab_size), d_model(d_model), 
    weight(weight), 
    BaseLayer(qdtype, device) {
    append_parameter(weight);
}

// === LINEAR ===

Linear::Linear(
    int d_in, int d_out,
    TPtr weight, TPtr bias, 
    DataType qdtype, DeviceType device
) : d_in(d_in), d_out(d_out), 
    weight(weight), bias(bias), 
    BaseLayer(qdtype, device) {
    append_parameter(weight);
    if (bias) {
        append_parameter(bias);
    }
}

// === RMSNORM ===

RMSNorm::RMSNorm(
    int dim, float eps, 
    TPtr weight, 
    DataType qdtype, DeviceType device
) : dim(dim), eps(eps), 
    weight(weight),
    BaseLayer(qdtype, device) {
    append_parameter(weight);
}

// === GQA ===

GQA::GQA(
    int block_idx, int d_model, size_t max_seq_len,
    int n_heads, int n_kv_heads, int d_head, int d_rotary,
    float eps, float freq_base,
    TPtr wq, TPtr wk, TPtr wv,
    TPtr wo, TPtr wq_norm, TPtr wk_norm,
    TPtr w_attnnorm, 
    DataType qdtype, DeviceType device
) : block_idx(block_idx), d_model(d_model), max_seq_len(max_seq_len), 
    n_heads(n_heads), n_kv_heads(n_kv_heads), d_head(d_head), d_rotary(d_rotary),
    eps(eps), freq_base(freq_base),
    wq(wq), wk(wk), wv(wv), 
    wo(wo), wq_norm(wq_norm), wk_norm(wk_norm),
    w_attnnorm(w_attnnorm),
    BaseLayer(qdtype, device),
    rope_table(
        compute_rope_table(max_seq_len, d_rotary, freq_base)
    ) {
    append_parameter(wq);
    append_parameter(wk);
    append_parameter(wv);
    append_parameter(wo);
    append_parameter(wq_norm);
    append_parameter(wk_norm);
    append_parameter(w_attnnorm);
}

std::vector<float> GQA::compute_rope_table(size_t max_seq_len, int d_rotary, float freq_base) {
    std::vector<float> table(max_seq_len * d_rotary);

    for (size_t pos=0; pos<max_seq_len; ++pos) {
        for (int i=0; i<d_rotary/2; ++i) {
            float freq = 1.0f / std::powf(freq_base, (2.0f * i)/d_rotary);
            table[pos*d_rotary + 2*i] = std::cosf(pos*freq);
            table[pos*d_rotary + 2*i+1] = std::sinf(pos*freq);
        }
    }

    return table;
}

// === MOE ===

MoE::MoE(
    int d_model, int d_ff, int n_experts, int n_active_experts, float eps,
    TPtr w_moenorm, TPtr w_router,
    TPtr ws_gate, TPtr ws_down, TPtr ws_up,
    DataType qdtype, DeviceType device
) : d_model(d_model), d_ff(d_ff), eps(eps), n_experts(n_experts), n_active_experts(n_active_experts),
    w_moenorm(w_moenorm), w_router(w_router), ws_gate(ws_gate), ws_down(ws_down), ws_up(ws_up),
    BaseLayer(qdtype, device)
     {
    append_parameter(w_moenorm);
    if (w_router) append_parameter(w_router); // only present in MoE models
    append_parameter(ws_gate);
    append_parameter(ws_down);
    append_parameter(ws_up);
}