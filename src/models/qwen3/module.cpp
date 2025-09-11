#include "minfer/models/qwen3/module.hpp"
#include "minfer/ops/cpu_ops.hpp"

#include <cassert>

// === TYPE CONVERSION UTILS ===
namespace {
    template<typename T>
    static float convert_to_float(T val) {
        if constexpr (std::is_same_v<T, float>) {
            return val;
        } else if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            return half_to_float(val);
        }
        // add more types as needed...
        else {
            static_assert(false && "Unsupported type for convert_to_float");
        }
    }
}

// === EMBED ===

Qwen3Embed::Qwen3Embed(
    size_t vocab_size, int d_model, 
    TPtr weight, 
    DataType qdtype, Device device
) : Embed(vocab_size, d_model, weight, qdtype, device) {

    set_read_bytes(weight->size_bytes / vocab_size); // only read a row of embed per forward pass
}

void Qwen3Embed::forward(std::shared_ptr<RunState> run_state) {
    // Device dispatch (GPU case not implemented yet)
    if (get_device() == Device::CPU) {
        switch (get_qdtype()) {
            case DataType::F32: Qwen3Embed::cpu_forward<float, float_tag>(run_state->x.get(), run_state->token_id); break;
            case DataType::F16: Qwen3Embed::cpu_forward<fp16_t, fp16_tag>(run_state->x.get(), run_state->token_id); break;
            case DataType::BF16: Qwen3Embed::cpu_forward<bf16_t, bf16_tag>(run_state->x.get(), run_state->token_id); break;
            default: assert(false && "Qwen3Embed has invalid or unsupported qdtype"); break;
        }
    } else {
        assert(false && "Qwen3Embed GPU support not implemented yet");
    }
}

template <typename WeightType, typename Tag>
void Qwen3Embed::cpu_forward(float* x_out, int token_id) {
    const WeightType* embed_vector = static_cast<WeightType*>(weight->data) + token_id * d_model;
    for (int i=0; i<d_model; ++i) {
        x_out[i] = convert_to_float(embed_vector[i]);
    }
}

// === LM HEAD ===

Qwen3LMHead::Qwen3LMHead(
    int d_in, int d_out,
    TPtr weight, TPtr bias, 
    DataType qdtype, Device device
) : Linear(d_in, d_out, weight, bias, qdtype, device) {
    set_read_bytes(weight->size_bytes + (bias ? bias->size_bytes : 0));
}

void Qwen3LMHead::forward(std::shared_ptr<RunState> run_state) {
    if (run_state->compute_logits) {
        if (get_device() == Device::CPU) {
            switch (get_qdtype()) {
                case DataType::F32: Qwen3LMHead::cpu_forward<float, float_tag>(run_state->logits.get(), run_state->x.get()); break;
                case DataType::F16: Qwen3LMHead::cpu_forward<fp16_t, fp16_tag>(run_state->logits.get(), run_state->x.get()); break;
                case DataType::BF16: Qwen3LMHead::cpu_forward<bf16_t, bf16_tag>(run_state->logits.get(), run_state->x.get()); break;
                default: assert(false && "Qwen3LMHead has invalid or unsupported qdtype"); break;
            }
        } else {
            assert(false && "Qwen3LMHead GPU support not implemented yet");
        }
    }
}

template <typename WeightType, typename Tag>
void Qwen3LMHead::cpu_forward(float* x_out, const float* x_in) {
    cpu::matmul<WeightType, Tag>(x_out, x_in, static_cast<WeightType*>(weight->data), d_out, d_in);
    if (bias) {
        for (int i=0; i<d_out; ++i) {
            x_out[i] += convert_to_float(static_cast<WeightType*>(bias->data)[i]);
        }
    }
}

// === FINAL RMS NORM ===

Qwen3FinalRMSNorm::Qwen3FinalRMSNorm(
    int dim, float eps, 
    TPtr weight,
    DataType qdtype, Device device
) : RMSNorm(dim, eps, weight, qdtype, device) {

    set_read_bytes(weight->size_bytes);
}

void Qwen3FinalRMSNorm::forward(std::shared_ptr<RunState> run_state) {
    // Device dispatch (GPU case not implemented yet)
    if (get_device() == Device::CPU) {
        switch (get_qdtype()) {
            case DataType::F32: Qwen3FinalRMSNorm::cpu_forward<float, float_tag>(run_state->x.get(), run_state->x.get()); break;
            default: assert(false && "Qwen3FinalRMSNorm has invalid or unsupported qdtype"); break;
        }
    } else {
        assert(false && "Qwen3FinalRMSNorm GPU support not implemented yet");
    }
}

template <typename WeightType, typename Tag>
void Qwen3FinalRMSNorm::cpu_forward(float* x_out, float* x_in) {
    cpu::rmsnorm(x_out, x_in, static_cast<WeightType*>(weight->data), dim, eps);
}

// === GQA ===

Qwen3GQA::Qwen3GQA(
    int block_idx, int d_model, size_t max_seq_len, 
    int n_heads, int n_kv_heads, int d_head, int d_rotary,
    float eps, float freq_base,
    TPtr wq, TPtr wk, TPtr wv,
    TPtr wo, TPtr wq_norm, TPtr wk_norm,
    TPtr w_attnnorm, 
    DataType qdtype, Device device
) : GQA(
        block_idx, d_model, max_seq_len, 
        n_heads, n_kv_heads, d_head, d_rotary,
        eps, freq_base, 
        wq, wk, wv, wo, wq_norm, wk_norm, w_attnnorm, 
        qdtype, device
    ) {
        set_read_bytes(
            wq->size_bytes + wk->size_bytes + wv->size_bytes + wo->size_bytes 
          + wq_norm->size_bytes + wk_norm->size_bytes + w_attnnorm->size_bytes
        );
    }

void Qwen3GQA::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == Device::CPU) {
        switch (get_qdtype()) {
            case DataType::F32: Qwen3GQA::cpu_forward<float, float_tag>(
                run_state->x.get(), run_state->xb.get(), 
                run_state->att_out.get(), run_state->att_scores.get(),
                run_state->q.get(), run_state->k.get(), run_state->v.get(),
                run_state->k_cache.get(), run_state->v_cache.get(),
                run_state->cur_pos
            ); break;

            case DataType::F16: Qwen3GQA::cpu_forward<fp16_t, fp16_tag>(
                run_state->x.get(), run_state->xb.get(), 
                run_state->att_out.get(), run_state->att_scores.get(),
                run_state->q.get(), run_state->k.get(), run_state->v.get(),
                run_state->k_cache.get(), run_state->v_cache.get(),
                run_state->cur_pos
            ); break;

            case DataType::BF16: Qwen3GQA::cpu_forward<bf16_t, bf16_tag>(
                run_state->x.get(), run_state->xb.get(), 
                run_state->att_out.get(), run_state->att_scores.get(),
                run_state->q.get(), run_state->k.get(), run_state->v.get(),
                run_state->k_cache.get(), run_state->v_cache.get(),
                run_state->cur_pos
            ); break;

            default: assert(false && "Qwen3GQA has invalid or unsupported qdtype"); break;
            
        }
        
    } else {
        assert(false && "Qwen3GQA GPU support not implemented yet");
    }
}

template <typename WeightType, typename Tag>
void Qwen3GQA::cpu_forward(
    float* x_in, float* x_norm, 
    float* att_out_buf, float* att_scores_buf, 
    float* q_buf, float* k_buf, float* v_buf,
    float* k_cache, float* v_cache,
    int cur_pos
) {
    cpu::rmsnorm(x_norm, x_in, static_cast<float*>(w_attnnorm->data), d_model, eps);
    
    cpu::matmul<WeightType,Tag>(q_buf, x_norm, static_cast<WeightType*>(wq->data), n_heads * d_head, d_model);
    cpu::matmul<WeightType,Tag>(k_buf, x_norm, static_cast<WeightType*>(wk->data), n_kv_heads * d_head, d_model);
    cpu::matmul<WeightType,Tag>(v_buf, x_norm, static_cast<WeightType*>(wv->data), n_kv_heads * d_head, d_model);


    // have to apply the rmsnorms per head
    for (int h=0; h<n_heads; ++h) {
        cpu::rmsnorm(q_buf + h*d_head, q_buf + h*d_head, static_cast<float*>(wq_norm->data), d_head, eps);
    }
    
    for (int h=0; h<n_kv_heads; ++h) {
        cpu::rmsnorm(k_buf + h*d_head, k_buf + h*d_head, static_cast<float*>(wk_norm->data), d_head, eps);
    }

    cpu::neox_rope(q_buf, q_buf, n_heads*d_head, d_head, d_rotary, freq_base, cur_pos);
    cpu::neox_rope(k_buf, k_buf, n_kv_heads*d_head, d_head, d_rotary, freq_base, cur_pos);

    // kv cache layout: [n_layers, max_seq_len, n_kv_heads, d_head]
    size_t cache_offset = block_idx * max_seq_len * n_kv_heads * d_head + cur_pos * n_kv_heads * d_head;
    for (int i=0; i < n_kv_heads * d_head; ++i) {
        k_cache[cache_offset+i] = k_buf[i];
        v_cache[cache_offset+i] = v_buf[i];
    }

    int heads_per_kv = n_heads/n_kv_heads;
    #pragma omp parallel for schedule(dynamic)
    for (int h=0; h<n_heads; ++h) {
        int kv_head = h/heads_per_kv;
        size_t kv_offset = block_idx*max_seq_len*n_kv_heads*d_head + kv_head*d_head;
        cpu::attn(
            att_scores_buf + h*max_seq_len,
            att_out_buf + h*d_head,
            q_buf + h*d_head,
            k_cache + kv_offset, // first pos kv_head
            v_cache + kv_offset, // see above
            cur_pos + 1,
            d_head,
            n_kv_heads * d_head
        );
    }

    cpu::matmul<WeightType,Tag>(x_norm, att_out_buf, static_cast<WeightType*>(wo->data), d_model, n_heads*d_head);

    for (int i=0; i<d_model; ++i) {
        x_in[i] += x_norm[i];
    }
}

// === MOE ===

Qwen3MoE::Qwen3MoE(
    int d_model, int d_ff, int n_experts, int n_active_experts, float eps,
    TPtr w_moenorm, TPtr w_router,
    TPtr ws_gate, TPtr ws_down, TPtr ws_up,
    DataType qdtype, Device device
) : MoE(
        d_model, d_ff, n_experts, n_active_experts, eps, 
        w_moenorm, w_router,
        ws_gate, ws_down, ws_up,
        qdtype, device
    ) {
        set_read_bytes(
          w_moenorm->size_bytes + (w_router ? w_router->size_bytes : 0)
        + (ws_gate->size_bytes / n_experts) * n_active_experts
        + (ws_down->size_bytes / n_experts) * n_active_experts
        + (ws_up->size_bytes / n_experts) * n_active_experts
        );
    }

void Qwen3MoE::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == Device::CPU) {
        switch(get_qdtype()) {
            case DataType::F32: Qwen3MoE::cpu_forward<float, float_tag>(
                run_state->x.get(), run_state->xb.get(), run_state->xb2.get(),
                run_state->hb.get(), run_state->hb2.get(),
                run_state->active_experts.get(), run_state->active_experts_scores.get(),
                run_state->active_experts_weights.get(), run_state->moe_scores.get()
            ); break;

            case DataType::F16: Qwen3MoE::cpu_forward<fp16_t, fp16_tag>(
                run_state->x.get(), run_state->xb.get(), run_state->xb2.get(),
                run_state->hb.get(), run_state->hb2.get(),
                run_state->active_experts.get(), run_state->active_experts_scores.get(),
                run_state->active_experts_weights.get(), run_state->moe_scores.get()
            ); break;

            case DataType::BF16: Qwen3MoE::cpu_forward<bf16_t, bf16_tag>(
                run_state->x.get(), run_state->xb.get(), run_state->xb2.get(),
                run_state->hb.get(), run_state->hb2.get(),
                run_state->active_experts.get(), run_state->active_experts_scores.get(),
                run_state->active_experts_weights.get(), run_state->moe_scores.get()
            ); break;

            default: assert(false && "Qwen3MoE has invalid or unsupported qdtype"); break;

        }
    } else {
        assert(false && "Qwen3MoE GPU support not implemented yet");
    }
}

template <typename WeightType, typename Tag>
void Qwen3MoE::cpu_forward(
    float* x_in, float* x_norm,
    float* exp_buf, float* gate_buf, float* up_buf,
    int* active_experts, float* active_experts_scores, 
    float* active_experts_weights, float* moe_scores
) {
    cpu::rmsnorm(x_norm, x_in, static_cast<float*>(w_moenorm->data), d_model, eps);

    if (n_experts == 0) {
        active_experts[0] = 0;
        active_experts_weights[0] = 1.0f;
    } else {
        cpu::route(
            x_norm, active_experts,
            active_experts_scores, active_experts_weights,
            moe_scores, static_cast<float*>(w_router->data),
            d_model, n_experts, n_active_experts
        );
    }

    int n = (n_experts > 0 ? n_active_experts : 1);
    for (int i=0; i < n; ++i) {
        int expert_idx = active_experts[i];

        const WeightType* w_gate = static_cast<WeightType*>(ws_gate->data) + expert_idx*d_ff*d_model;
        const WeightType* w_up = static_cast<WeightType*>(ws_up->data) + expert_idx*d_ff*d_model;
        const WeightType* w_down = static_cast<WeightType*>(ws_down->data) + expert_idx*d_model*d_ff;
            
        cpu::swiglu<WeightType,Tag>(x_norm, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);

        for (int j=0; j<d_model; ++j) {
            x_in[j] += exp_buf[j]*active_experts_weights[i];
        }
    }
}

// explicit instantiations
template void Qwen3Embed::cpu_forward<float, float_tag>(float*, int);
template void Qwen3Embed::cpu_forward<fp16_t, fp16_tag>(float*, int);
template void Qwen3Embed::cpu_forward<bf16_t, bf16_tag>(float*, int);

template void Qwen3LMHead::cpu_forward<float, float_tag>(float*, const float*);
template void Qwen3LMHead::cpu_forward<fp16_t, fp16_tag>(float*, const float*);
template void Qwen3LMHead::cpu_forward<bf16_t, bf16_tag>(float*, const float*);

template void Qwen3FinalRMSNorm::cpu_forward<float, float_tag>(float* x_out, float* x_in);

template void Qwen3GQA::cpu_forward<float, float_tag>(float*, float*, float*, float*, float*, float*, float*, float*, float*, int);
template void Qwen3GQA::cpu_forward<fp16_t, fp16_tag>(float*, float*, float*, float*, float*, float*, float*, float*, float*, int);
template void Qwen3GQA::cpu_forward<bf16_t, bf16_tag>(float*, float*, float*, float*, float*, float*, float*, float*, float*, int);

template void Qwen3MoE::cpu_forward<float, float_tag>(float*, float*, float*, float*, float*, int*, float*, float*, float*);
template void Qwen3MoE::cpu_forward<fp16_t, fp16_tag>(float*, float*, float*, float*, float*, int*, float*, float*, float*);
template void Qwen3MoE::cpu_forward<bf16_t, bf16_tag>(float*, float*, float*, float*, float*, int*, float*, float*, float*);