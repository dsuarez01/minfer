#include "minfer/base/config.hpp"
#include "minfer/base/module.hpp"
#include "minfer/base/tensor.hpp"

#include <cassert>

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
    DeviceType device
) : vocab_size(vocab_size), d_model(d_model), 
    weight(weight), 
    BaseLayer(device) {
    append_parameter(weight);
    set_read_bytes(weight->size_bytes / vocab_size); // only read a row of embed per forward pass
}

// === LINEAR ===

Linear::Linear(
    int d_in, int d_out,
    TPtr weight, TPtr bias, 
    DeviceType device
) : d_in(d_in), d_out(d_out), 
    weight(weight), bias(bias), 
    BaseLayer(device) {
    assert((!bias || bias->dtype == DataType::F32) && "Only FP32 bias weight supported in linear");

    append_parameter(weight);
    if (bias) {
        append_parameter(bias);
    }

    set_read_bytes(weight->size_bytes + (bias ? bias->size_bytes : 0));
}

// === RMSNORM ===

RMSNorm::RMSNorm(
    int dim, float eps, 
    TPtr weight, 
    DeviceType device
) : dim(dim), eps(eps), 
    weight(weight),
    BaseLayer(device) {
    assert(weight->dtype == DataType::F32 && "Dtype of this tensor should be F32");
    append_parameter(weight);
    set_read_bytes(weight->size_bytes);
}

// === GQA ===

GQA::GQA(
    int block_idx, int d_model, size_t max_seq_len,
    int n_heads, int n_kv_heads, int d_head, int d_rotary,
    float eps, float freq_base,
    TPtr wq, TPtr wk, TPtr wv,
    TPtr wo, TPtr wq_norm, TPtr wk_norm,
    TPtr w_attnnorm, 
    DeviceType device
) : block_idx(block_idx), d_model(d_model), max_seq_len(max_seq_len), 
    n_heads(n_heads), n_kv_heads(n_kv_heads), d_head(d_head), d_rotary(d_rotary),
    eps(eps), freq_base(freq_base),
    wq(wq), wk(wk), wv(wv), 
    wo(wo), wq_norm(wq_norm), wk_norm(wk_norm),
    w_attnnorm(w_attnnorm),
    BaseLayer(device) 
{
    assert(
        wq_norm->dtype == wk_norm->dtype &&
        wk_norm->dtype == w_attnnorm->dtype &&
        w_attnnorm->dtype == DataType::F32 &&
        "Dtypes of these tensors should be F32"
    );

    assert(
        wq->dtype == wk->dtype && 
        wk->dtype == wv->dtype && 
        wv->dtype == wo->dtype &&
        "Dtypes of these tensors should be identical"
    );

    append_parameter(wq);
    append_parameter(wk);
    append_parameter(wv);
    append_parameter(wo);
    append_parameter(wq_norm);
    append_parameter(wk_norm);
    append_parameter(w_attnnorm);

    set_read_bytes(
        wq->size_bytes + wk->size_bytes + wv->size_bytes + wo->size_bytes 
        + wq_norm->size_bytes + wk_norm->size_bytes + w_attnnorm->size_bytes
    );
}

// === MOE ===

MoE::MoE(
    int d_model, int d_ff, int n_experts, int n_active_experts, float eps,
    TPtr w_moenorm, TPtr w_router,
    TPtr ws_gate, TPtr ws_down, TPtr ws_up,
    DeviceType device
) : d_model(d_model), d_ff(d_ff), eps(eps), n_experts(n_experts), n_active_experts(n_active_experts),
    w_moenorm(w_moenorm), w_router(w_router), ws_gate(ws_gate), ws_down(ws_down), ws_up(ws_up),
    BaseLayer(device)
{

    assert(
        w_moenorm->dtype == DataType::F32 &&
        (!w_router || w_router->dtype == DataType::F32) &&
        "Dtypes of these tensors should be F32"
    );

    assert(
        ws_gate->dtype == ws_down->dtype && 
        ws_down->dtype == ws_up->dtype && 
        "Dtypes of these tensors should be identical"
    );

    append_parameter(w_moenorm);
    if (w_router) append_parameter(w_router); // only present in MoE models
    append_parameter(ws_gate);
    append_parameter(ws_down);
    append_parameter(ws_up);

    set_read_bytes(
        w_moenorm->size_bytes + (w_router ? w_router->size_bytes : 0)
      + (ws_gate->size_bytes / n_experts) * n_active_experts
      + (ws_down->size_bytes / n_experts) * n_active_experts
      + (ws_up->size_bytes / n_experts) * n_active_experts
    );
}