#include "minfer/models/qwen3/module.hpp"
#include "minfer/ops/kernels.hpp"
#include "minfer/config/metal_config.hpp"

#include <cassert>

// === TYPE CONVERSION UTILS ===
namespace {
    template<typename T, typename Tag>
    static float convert_to_float(T val) {
        if constexpr (std::is_same_v<T, float>) {
            return val;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            if constexpr (std::is_same_v<Tag, fp16_tag>) {
                return fp16_to_float(val);
            } else if constexpr (std::is_same_v<Tag, bf16_tag>) {
                return bf16_to_float(val);
            } else {
                static_assert(false && "Unsupported tag for uint16_t conversion");
            }
        } else {
            static_assert(false && "Unsupported type in convert_to_float");
        }
    }
}

// === EMBED ===

Qwen3Embed::Qwen3Embed(
    size_t vocab_size, int d_model, 
    TPtr weight, 
    DataType qdtype, DeviceType device
) : Embed(vocab_size, d_model, weight, qdtype, device) {

    set_read_bytes(weight->size_bytes / vocab_size); // only read a row of embed per forward pass
}

void Qwen3Embed::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == DeviceType::CPU) {
        switch (get_qdtype()) {
            case DataType::F32: Qwen3Embed::cpu_forward<float, float_tag>(run_state->x.get(), run_state->token_id); break;
            case DataType::F16: Qwen3Embed::cpu_forward<fp16_t, fp16_tag>(run_state->x.get(), run_state->token_id); break;
            case DataType::BF16: Qwen3Embed::cpu_forward<bf16_t, bf16_tag>(run_state->x.get(), run_state->token_id); break;
            default: assert(false && "Qwen3Embed has invalid or unsupported qdtype"); break;
        }
    } else if (get_device() == DeviceType::METAL) {
        Qwen3Embed::metal_forward(run_state->x.get(), run_state->token_id);
    } else {
        assert(false && "Qwen3Embed support for this device not implemented yet");
    }
}

template <typename WeightType, typename Tag>
void Qwen3Embed::cpu_forward(float* x_out, int token_id) {
    const WeightType* embed_vector = static_cast<WeightType*>(weight->data) + token_id * d_model;
    for (int i=0; i<d_model; ++i) {
        x_out[i] = convert_to_float<WeightType, Tag>(embed_vector[i]);
    }
}

void Qwen3Embed::metal_forward(float* x_out, int token_id) {

    std::string embed_name = "embed" + dtype_kernel_suffix(get_qdtype());
    int token_offset = token_id*d_model;

    MetalManager::dispatch1d(
        embed_name.c_str(),
        d_model / 32, 32,
        &token_offset, sizeof(int),
        (void*[]){x_out, weight->data}, 2
    );
}

// === LM HEAD ===

Qwen3LMHead::Qwen3LMHead(
    int d_in, int d_out,
    TPtr weight, TPtr bias, 
    DataType qdtype, DeviceType device
) : Linear(d_in, d_out, weight, bias, qdtype, device) {
    set_read_bytes(weight->size_bytes + (bias ? bias->size_bytes : 0));
}

void Qwen3LMHead::forward(std::shared_ptr<RunState> run_state) {
    if (run_state->compute_logits) {
        if (get_device() == DeviceType::CPU) {
            switch (get_qdtype()) {
                case DataType::F32: Qwen3LMHead::cpu_forward<float, float_tag>(run_state->logits.get(), run_state->x.get()); break;
                case DataType::F16: Qwen3LMHead::cpu_forward<fp16_t, fp16_tag>(run_state->logits.get(), run_state->x.get()); break;
                case DataType::BF16: Qwen3LMHead::cpu_forward<bf16_t, bf16_tag>(run_state->logits.get(), run_state->x.get()); break;
                default: assert(false && "Qwen3LMHead has invalid or unsupported qdtype"); break;
            }
        } else if (get_device() == DeviceType::METAL) {
            Qwen3LMHead::metal_forward(run_state->logits.get(), run_state->x.get());
        } else {
            assert(false && "Qwen3LMHead support for this device not implemented yet");
        }
    }
}

template <typename WeightType, typename Tag>
void Qwen3LMHead::cpu_forward(float* x_out, const float* x_in) {
    cpu::matmul<WeightType, Tag>(x_out, x_in, static_cast<WeightType*>(weight->data), d_out, d_in);
    if (bias) {
        for (int i=0; i<d_out; ++i) {
            x_out[i] += convert_to_float<WeightType, Tag>(static_cast<WeightType*>(bias->data)[i]);
        }
    }
}

void Qwen3LMHead::metal_forward(float* x_out, const float* x_in) {

    std::string lin_proj_name = "linear_proj" + dtype_kernel_suffix(get_qdtype());

    struct { size_t weight_offset; int d_in; } args = { 0, d_in };

    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        d_out, 32,
        &args, sizeof(args),
        (void*[]){weight->data, const_cast<float*>(x_in), x_out}, 3
    );
    
    if (bias) {
        size_t n_thrgps = (d_out+1023) / 1024;
        MetalManager::dispatch1d(
            "resadd",
            n_thrgps, 1024,
            nullptr, 0,
            (void*[]){x_out, bias->data}, 2
        );
    }
}

// === FINAL RMS NORM ===

Qwen3FinalRMSNorm::Qwen3FinalRMSNorm(
    int dim, float eps, 
    TPtr weight,
    DataType qdtype, DeviceType device
) : RMSNorm(dim, eps, weight, qdtype, device) {

    set_read_bytes(weight->size_bytes);
}

void Qwen3FinalRMSNorm::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == DeviceType::CPU) {
        switch (get_qdtype()) {
            case DataType::F32: Qwen3FinalRMSNorm::cpu_forward<float, float_tag>(run_state->x.get(), run_state->x.get()); break;
            default: assert(false && "Qwen3FinalRMSNorm has invalid or unsupported qdtype"); break;
        }
    } else if (get_device() == DeviceType::METAL) {
        Qwen3FinalRMSNorm::metal_forward(run_state->x.get(), run_state->x.get());
    } else {
        assert(false && "Qwen3FinalRMSNorm support for this device not implemented yet");
    }
}

template <typename WeightType, typename Tag>
void Qwen3FinalRMSNorm::cpu_forward(float* x_out, float* x_in) {
    cpu::rmsnorm(x_out, x_in, static_cast<WeightType*>(weight->data), dim, eps);
}

void Qwen3FinalRMSNorm::metal_forward(float* x_out, float* x_in) {
    
    struct { int dim; float eps; int stride; } params = {dim, eps, 0};
    
    MetalManager::dispatch1d(
        "rmsnorm",
        1, 1024,
        &params, sizeof(params),
        (void*[]){weight->data, x_in, x_out}, 3
    );
}

// === GQA ===

Qwen3GQA::Qwen3GQA(
    int block_idx, int d_model, size_t max_seq_len, 
    int n_heads, int n_kv_heads, int d_head, int d_rotary,
    float eps, float freq_base,
    TPtr wq, TPtr wk, TPtr wv,
    TPtr wo, TPtr wq_norm, TPtr wk_norm,
    TPtr w_attnnorm, 
    DataType qdtype, DeviceType device
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
    if (get_device() == DeviceType::CPU) {
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
        
    } else if (get_device() == DeviceType::METAL) {
        Qwen3GQA::metal_forward(
            run_state->x.get(), run_state->xb.get(), 
            run_state->att_out.get(), run_state->att_scores.get(),
            run_state->q.get(), run_state->k.get(), run_state->v.get(),
            run_state->k_cache.get(), run_state->v_cache.get(),
            run_state->cur_pos
        );
    } else {
        assert(false && "Qwen3GQA support for this device not implemented yet");
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

    cpu::neox_rope(q_buf, q_buf, n_heads*d_head, d_head, d_rotary, freq_base, cur_pos, rope_table);
    cpu::neox_rope(k_buf, k_buf, n_kv_heads*d_head, d_head, d_rotary, freq_base, cur_pos, rope_table);

    // kv cache layout: [n_layers, max_seq_len, n_kv_heads, d_head]
    size_t cache_offset = block_idx * max_seq_len * n_kv_heads * d_head + cur_pos * n_kv_heads * d_head;
    for (int i=0; i < n_kv_heads * d_head; ++i) {
        k_cache[cache_offset+i] = k_buf[i];
        v_cache[cache_offset+i] = v_buf[i];
    }

    int heads_per_kv = n_heads/n_kv_heads;
    
    #pragma omp parallel for
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

void Qwen3GQA::metal_forward(
    float* x_in, float* x_norm, 
    float* att_out_buf, float* att_scores_buf, 
    float* q_buf, float* k_buf, float* v_buf,
    float* k_cache, float* v_cache,
    int cur_pos
) {
    
    std::string lin_proj_name = "linear_proj" + dtype_kernel_suffix(get_qdtype());

    int q_dim = n_heads * d_head;
    int kv_dim = n_kv_heads * d_head;
    int kv_len = cur_pos + 1;
    int heads_per_kv = n_heads / n_kv_heads;
    
    // 1. pre-RMSNorm
    struct { int dim; float eps; int stride; } norm_params = {d_model, eps, 0};
    MetalManager::dispatch1d(
        "rmsnorm",
        1, 1024,
        &norm_params, sizeof(norm_params),
        (void*[]){w_attnnorm->data, x_in, x_norm}, 3
    );

    // 2-4. QKV projs

    // for clarity, but otherwise redundant
    struct { size_t offset; int d_in; } q_proj_args = { 0, d_model };
    struct { size_t offset; int d_in; } k_proj_args = { 0, d_model };
    struct { size_t offset; int d_in; } v_proj_args = { 0, d_model };

    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        q_dim, 32,
        &q_proj_args, sizeof(q_proj_args),
        (void*[]){wq->data, x_norm, q_buf}, 3
    );

    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        kv_dim, 32,
        &k_proj_args, sizeof(k_proj_args),
        (void*[]){wk->data, x_norm, k_buf}, 3
    );
    
    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        kv_dim, 32,
        &v_proj_args, sizeof(v_proj_args),
        (void*[]){wv->data, x_norm, v_buf}, 3
    );

    // 5-6. rmsnorm across heads for Q,K
    struct { int dim; float eps; int stride; } head_norm = {d_head, eps, d_head};
    MetalManager::dispatch1d(
        "rmsnorm",
        n_heads, 1024,
        &head_norm, sizeof(head_norm),
        (void*[]){wq_norm->data, q_buf, q_buf}, 3
    );
    
    MetalManager::dispatch1d(
        "rmsnorm",
        n_kv_heads, 1024,
        &head_norm, sizeof(head_norm),
        (void*[]){wk_norm->data, k_buf, k_buf}, 3
    );

    // 7-8. RoPE

    // 7. RoPE on Q
    struct { int d_rotary; int d_head; float freq_base; int pos; } rope_params = {d_rotary, d_head, freq_base, cur_pos};
    size_t rope_thrgps = (d_rotary/2 + 31) / 32;
    
    MetalManager::dispatch2d(
        "neox_rope",
        rope_thrgps, n_heads,  // X=pair thrgps, Y=heads
        32,
        &rope_params, sizeof(rope_params),
        (void*[]){q_buf}, 1
    );

    // 8. RoPE on K
    MetalManager::dispatch2d(
        "neox_rope",
        rope_thrgps, n_kv_heads,  // Y = n_kv_heads instead
        32,
        &rope_params, sizeof(rope_params),
        (void*[]){k_buf}, 1
    );

    // 9. writing to K,V caches
    struct { int layer_idx; int cur_pos; int seq_len; int n_kv_heads; int d_head; } cache_params = {
        block_idx, cur_pos, (int)max_seq_len, n_kv_heads, d_head
    };
    size_t cache_thrgps = (kv_dim+1023) / 1024;
    MetalManager::dispatch1d(
        "write_kv_cache",
        cache_thrgps, 1024,
        &cache_params, sizeof(cache_params),
        (void*[]){k_buf, v_buf, k_cache, v_cache}, 4
    );

    // 10-12. attn scoring, mixing
    
    // attn scoring    
    struct { size_t loff; int d_head; int kv_dim; int kv_mul; int kv_len; int seq_len; } score_params = {
        block_idx * max_seq_len * kv_dim, d_head, kv_dim, heads_per_kv, kv_len, (int)max_seq_len
    };

    MetalManager::dispatch2d(
        "attn_score",
        kv_len, n_heads,  // X=posns, Y=heads
        32,
        &score_params, sizeof(score_params),
        (void*[]){q_buf, k_cache, att_scores_buf}, 3
    );

    // softmax
    struct { int dim; int stride; } softmax_params = {kv_len, (int)max_seq_len};
    MetalManager::dispatch1d(
        "softmax",
        n_heads, 1024,
        &softmax_params, sizeof(softmax_params),
        (void*[]){att_scores_buf, att_scores_buf}, 2
    );

    // attn mixing
    struct { size_t loff; int seq_len; int kv_len; int d_head; int kv_dim; int kv_mul; } out_params = {
        block_idx * max_seq_len * kv_dim, (int)max_seq_len, kv_len, d_head, kv_dim, heads_per_kv
    };

    MetalManager::dispatch2d(
        "attn_out",
        d_head, n_heads, // X=head dim, Y=heads
        32,
        &out_params, sizeof(out_params),
        (void*[]){att_scores_buf, v_cache, att_out_buf}, 3
    );

    struct { size_t offset; int d_in; } output_args = { 0, q_dim };

    // 13. output proj
    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        d_model, 32,
        &output_args, sizeof(output_args),
        (void*[]){wo->data, att_out_buf, x_norm}, 3
    );

    // 14. residual
    size_t res_thrgps = (d_model+1023) / 1024;
    MetalManager::dispatch1d(
        "resadd",
        res_thrgps, 1024,
        nullptr, 0,
        (void*[]){x_in, x_norm}, 2
    );
}

// === MOE ===

Qwen3MoE::Qwen3MoE(
    int d_model, int d_ff, int n_experts, int n_active_experts, float eps,
    TPtr w_moenorm, TPtr w_router,
    TPtr ws_gate, TPtr ws_down, TPtr ws_up,
    DataType qdtype, DeviceType device
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
    if (get_device() == DeviceType::CPU) {
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
    } else if (get_device() == DeviceType::METAL) {
        Qwen3MoE::metal_forward(
            run_state->x.get(), run_state->xb.get(), run_state->xb2.get(),
            run_state->hb.get(), run_state->hb2.get(),
            run_state->active_experts.get(), run_state->active_experts_scores.get(),
            run_state->active_experts_weights.get(), run_state->moe_scores.get()
        );
    } else {
        assert(false && "Qwen3MoE support for this device not implemented yet");
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

void Qwen3MoE::metal_forward(
    float* x_in, float* x_norm,
    float* exp_buf, float* gate_buf, float* up_buf,
    int* active_experts, float* active_experts_scores, 
    float* active_experts_weights, float* moe_scores
) {

    DataType qdtype = get_qdtype();
    std::string lin_proj_name = "linear_proj" + dtype_kernel_suffix(qdtype);

    // 1. rmsnorm
    struct { int dim; float eps; int stride; } norm_params = {d_model, eps, 0};
    MetalManager::dispatch1d(
        "rmsnorm",
        1, 1024,
        &norm_params, sizeof(norm_params),
        (void*[]){w_moenorm->data, x_in, x_norm}, 3
    );

    // if MoE model
    if (n_experts > 0) {

        // 2. router
        struct { size_t weight_offset; int d_in; } router_args = { 0, d_model };

        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            n_experts, 32,
            &router_args, sizeof(router_args),
            (void*[]){w_router->data, x_norm, moe_scores}, 3
        );

        // 3. top-k
        struct { int n_experts; int k; } topk_params = {n_experts, n_active_experts};
        MetalManager::dispatch1d(
            "moe_topk",
            1, 32,
            &topk_params, sizeof(topk_params),
            (void*[]){moe_scores, active_experts, active_experts_weights}, 3
        );

        // 4. softmax
        struct { int dim; int stride; } softmax_params = {n_active_experts, 0};
        MetalManager::dispatch1d(
            "softmax",
            1, 1024,
            &softmax_params, sizeof(softmax_params),
            (void*[]){active_experts_weights, active_experts_weights}, 2
        );
    }

    int* active_experts_cpu = (int*)MetalManager::buf_contents(active_experts);
    float* active_experts_weights_cpu = (float*)MetalManager::buf_contents(active_experts_weights);

    // 5. process experts
    int n = (n_experts > 0 ? n_active_experts : 1);
    for (int i=0; i<n; ++i) {
        int expert_idx = active_experts_cpu[i];

        // for clarity, but otherwise redundant
        struct { size_t weight_offset; int d_in; } gate_args = { (size_t) expert_idx*d_ff*d_model, d_model };
        struct { size_t weight_offset; int d_in; } up_args = { (size_t) expert_idx*d_ff*d_model, d_model };
        struct { size_t weight_offset; int d_in; } down_args = { (size_t) expert_idx*d_model*d_ff, d_ff };

        // gate proj
        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            d_ff, 32,
            &gate_args, sizeof(gate_args),
            (void*[]){ws_gate->data, x_norm, gate_buf}, 3
        );

        // up proj
        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            d_ff, 32,
            &up_args, sizeof(up_args),
            (void*[]){ws_up->data, x_norm, up_buf}, 3
        );

        // silu+mul
        size_t silu_thrgps = (d_ff+1023) / 1024;
        MetalManager::dispatch1d(
            "silu_mul",
            silu_thrgps, 1024,
            nullptr, 0,
            (void*[]){gate_buf, up_buf}, 2
        );

        // down proj
        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            d_model, 32,
            &down_args, sizeof(down_args),
            (void*[]){ws_down->data, gate_buf, exp_buf}, 3
        );

        // weighted resadd
        float weight = active_experts_weights_cpu[i];
        size_t res_thrgps = (d_model+1023) / 1024;
        MetalManager::dispatch1d(
            "weight_resadd",
            res_thrgps, 1024,
            &weight, sizeof(float),
            (void*[]){x_in, exp_buf}, 2
        );
    }
}

// explicit instantiations
// CPU
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