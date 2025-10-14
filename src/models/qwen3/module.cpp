#include "minfer/models/qwen3/module.hpp"
#include "minfer/ops/kernels.hpp"
#include "minfer/base/tensor.hpp"
#include "minfer/base/types.hpp"

#include "minfer/interfaces/metal_interface.hpp"

#include <cassert>
#include <iostream>

// === EMBED ===

Qwen3Embed::Qwen3Embed(
    size_t vocab_size, int d_model, 
    TPtr weight,
    DeviceType device
) : Embed(vocab_size, d_model, weight, device) {}

void Qwen3Embed::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == DeviceType::CPU) {
        Qwen3Embed::cpu_forward(std::get<float*>(run_state->x), run_state->token_id);
    } else if (get_device() == DeviceType::METAL) {
        Qwen3Embed::metal_forward(std::get<MTL::Buffer*>(run_state->x), run_state->token_id);
    } else {
        assert(false && "Qwen3Embed support for this device not implemented yet");
    }
}

void Qwen3Embed::cpu_forward(float* x_out, uint32_t token_id) {
    embed(x_out, weight, token_id, d_model);
}

void Qwen3Embed::metal_forward(MTL::Buffer* x_out, int token_id) {

    std::string embed_name = "embed" + dtype_kernel_suffix(weight->dtype);
    int token_offset = token_id*d_model;

    MetalManager::dispatch1d(
        embed_name.c_str(),
        d_model / 32, 32,
        &token_offset, sizeof(int),
        (MTL::Buffer*[]){x_out, weight->metal_buf()}, 2
    );
}

// === LM HEAD ===

Qwen3LMHead::Qwen3LMHead(
    int d_in, int d_out,
    TPtr weight, TPtr bias, 
    DeviceType device
) : Linear(d_in, d_out, weight, bias, device) {}

void Qwen3LMHead::forward(std::shared_ptr<RunState> run_state) {
    if (run_state->compute_logits) {
        if (get_device() == DeviceType::CPU) {
            Qwen3LMHead::cpu_forward(std::get<float*>(run_state->logits), std::get<float*>(run_state->x));
        } else if (get_device() == DeviceType::METAL) {
            Qwen3LMHead::metal_forward(std::get<MTL::Buffer*>(run_state->logits), std::get<MTL::Buffer*>(run_state->x));
        } else {
            assert(false && "Qwen3LMHead support for this device not implemented yet");
        }
    }
}

void Qwen3LMHead::cpu_forward(float* x_out, float* x_in) {
    matmul(x_out, x_in, weight, 0, d_out, d_in);
    
    if (bias) {
        auto& bias_view = bias->cpu_typed_view<DataType::F32>();
        const float* b = bias_view.ptr();
        for (int i=0; i<d_out; ++i) {
            x_out[i] += b[i];
        }
    }
}

void Qwen3LMHead::metal_forward(MTL::Buffer* x_out, MTL::Buffer* x_in) {

    std::string lin_proj_name = "linear_proj" + dtype_kernel_suffix(weight->dtype);

    struct { size_t weight_offset; int d_in; } args = { 0, d_in };

    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        d_out, 32,
        &args, sizeof(args),
        (MTL::Buffer*[]){weight->metal_buf(), x_in, x_out}, 3
    );
    
    if (bias) {
        size_t n_thrgps = (d_out+1023) / 1024;
        MetalManager::dispatch1d(
            "resadd",
            n_thrgps, 1024,
            nullptr, 0,
            (MTL::Buffer*[]){x_out, bias->metal_buf()}, 2
        );
    }
}

// === FINAL RMS NORM ===

Qwen3FinalRMSNorm::Qwen3FinalRMSNorm(
    int dim, float eps, 
    TPtr weight,
    DeviceType device
) : RMSNorm(dim, eps, weight, device) {}

void Qwen3FinalRMSNorm::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == DeviceType::CPU) {
        Qwen3FinalRMSNorm::cpu_forward(std::get<float*>(run_state->x), std::get<float*>(run_state->x));
    } else if (get_device() == DeviceType::METAL) {
        Qwen3FinalRMSNorm::metal_forward(std::get<MTL::Buffer*>(run_state->x), std::get<MTL::Buffer*>(run_state->x));
    } else {
        assert(false && "Qwen3FinalRMSNorm support for this device not implemented yet");
    }
}

void Qwen3FinalRMSNorm::cpu_forward(float* x_out, float* x_in) {
    auto& weight_view = weight->cpu_typed_view<DataType::F32>();
    rmsnorm(x_out, x_in, weight_view, dim, eps);
}

void Qwen3FinalRMSNorm::metal_forward(MTL::Buffer* x_out, MTL::Buffer* x_in) {
    
    struct { int dim; float eps; int stride; } params = {dim, eps, 0};
    
    MetalManager::dispatch1d(
        "rmsnorm",
        1, 1024,
        &params, sizeof(params),
        (MTL::Buffer*[]){weight->metal_buf(), x_in, x_out}, 3
    );
}

// === GQA ===

Qwen3GQA::Qwen3GQA(
    int block_idx, int d_model, int n_heads, int n_kv_heads, int d_head, int d_rotary,
    size_t max_seq_len,
    float eps, float freq_base,
    TPtr wq, TPtr wk, TPtr wv,
    TPtr wo, TPtr wq_norm, TPtr wk_norm,
    TPtr w_attnnorm, 
    DeviceType device
) : GQA(
        block_idx, d_model, n_heads, n_kv_heads, d_head, d_rotary,
        max_seq_len,
        eps, freq_base, 
        wq, wk, wv, wo, wq_norm, wk_norm, w_attnnorm, 
        device
    ) {}

void Qwen3GQA::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == DeviceType::CPU) {
        Qwen3GQA::cpu_forward(
            std::get<float*>(run_state->x), std::get<float*>(run_state->xb), 
            std::get<float*>(run_state->att_out), std::get<float*>(run_state->att_scores),
            std::get<float*>(run_state->q), std::get<float*>(run_state->k), std::get<float*>(run_state->v),
            std::get<float*>(run_state->k_cache), std::get<float*>(run_state->v_cache),
            run_state->cur_pos
        );
    } else if (get_device() == DeviceType::METAL) {
        Qwen3GQA::metal_forward(
            std::get<MTL::Buffer*>(run_state->x), std::get<MTL::Buffer*>(run_state->xb), 
            std::get<MTL::Buffer*>(run_state->att_out), std::get<MTL::Buffer*>(run_state->att_scores),
            std::get<MTL::Buffer*>(run_state->q), std::get<MTL::Buffer*>(run_state->k), std::get<MTL::Buffer*>(run_state->v),
            std::get<MTL::Buffer*>(run_state->k_cache), std::get<MTL::Buffer*>(run_state->v_cache),
            run_state->cur_pos
        );
    } else {
        assert(false && "Qwen3GQA support for this device not implemented yet");
    }
}

void Qwen3GQA::cpu_forward(
    float* x_in, float* x_norm, 
    float* att_out_buf, float* att_scores_buf, 
    float* q_buf, float* k_buf, float* v_buf,
    float* k_cache, float* v_cache,
    int cur_pos
) {
    rmsnorm(x_norm, x_in, w_attnnorm->cpu_typed_view<DataType::F32>(), d_model, eps);

    matmul(q_buf, x_norm, wq, 0, n_heads * d_head, d_model);
    matmul(k_buf, x_norm, wk, 0, n_kv_heads * d_head, d_model);
    matmul(v_buf, x_norm, wv, 0, n_kv_heads * d_head, d_model);

    // have to apply the rmsnorms per head
    for (int h=0; h<n_heads; ++h) {
        rmsnorm(q_buf + h*d_head, q_buf + h*d_head, wq_norm->cpu_typed_view<DataType::F32>(), d_head, eps);
    }
    
    for (int h=0; h<n_kv_heads; ++h) {
        rmsnorm(k_buf + h*d_head, k_buf + h*d_head, wk_norm->cpu_typed_view<DataType::F32>(), d_head, eps);
    }

    neox_rope(q_buf, q_buf, n_heads*d_head, d_head, d_rotary, freq_base, cur_pos);
    neox_rope(k_buf, k_buf, n_kv_heads*d_head, d_head, d_rotary, freq_base, cur_pos);

    // kv cache layout: [n_layers, max_seq_len, n_kv_heads, d_head]
    size_t cache_offset = block_idx * max_seq_len * n_kv_heads * d_head + cur_pos * n_kv_heads * d_head;
    for (int i=0; i < n_kv_heads*d_head; ++i) {
        k_cache[cache_offset+i] = k_buf[i];
        v_cache[cache_offset+i] = v_buf[i];
    }

    int heads_per_kv = n_heads/n_kv_heads;
    
    #pragma omp parallel for
    for (int h=0; h<n_heads; ++h) {
        int kv_head = h/heads_per_kv;
        size_t kv_offset = block_idx*max_seq_len*n_kv_heads*d_head + kv_head*d_head;
        attn(
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

    matmul(x_norm, att_out_buf, wo, 0, d_model, n_heads*d_head);

    for (int i=0; i<d_model; ++i) {
        x_in[i] += x_norm[i];
    }
}

void Qwen3GQA::metal_forward(
    MTL::Buffer* x_in, MTL::Buffer* x_norm, 
    MTL::Buffer* att_out_buf, MTL::Buffer* att_scores_buf, 
    MTL::Buffer* q_buf, MTL::Buffer* k_buf, MTL::Buffer* v_buf,
    MTL::Buffer* k_cache, MTL::Buffer* v_cache,
    int cur_pos
) {
    
    // assert matching dtypes of qkv, output tensors in init.
    std::string lin_proj_name = "linear_proj" + dtype_kernel_suffix(wq->dtype);

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
        (MTL::Buffer*[]){w_attnnorm->metal_buf(), x_in, x_norm}, 3
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
        (MTL::Buffer*[]){wq->metal_buf(), x_norm, q_buf}, 3
    );

    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        kv_dim, 32,
        &k_proj_args, sizeof(k_proj_args),
        (MTL::Buffer*[]){wk->metal_buf(), x_norm, k_buf}, 3
    );
    
    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        kv_dim, 32,
        &v_proj_args, sizeof(v_proj_args),
        (MTL::Buffer*[]){wv->metal_buf(), x_norm, v_buf}, 3
    );

    // 5-6. rmsnorm across heads for Q,K
    struct { int dim; float eps; int stride; } head_norm = {d_head, eps, d_head};
    MetalManager::dispatch1d(
        "rmsnorm",
        n_heads, 1024,
        &head_norm, sizeof(head_norm),
        (MTL::Buffer*[]){wq_norm->metal_buf(), q_buf, q_buf}, 3
    );
    
    MetalManager::dispatch1d(
        "rmsnorm",
        n_kv_heads, 1024,
        &head_norm, sizeof(head_norm),
        (MTL::Buffer*[]){wk_norm->metal_buf(), k_buf, k_buf}, 3
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
        (MTL::Buffer*[]){q_buf}, 1
    );

    // 8. RoPE on K
    MetalManager::dispatch2d(
        "neox_rope",
        rope_thrgps, n_kv_heads,  // Y = n_kv_heads instead
        32,
        &rope_params, sizeof(rope_params),
        (MTL::Buffer*[]){k_buf}, 1
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
        (MTL::Buffer*[]){k_buf, v_buf, k_cache, v_cache}, 4
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
        (MTL::Buffer*[]){q_buf, k_cache, att_scores_buf}, 3
    );

    // softmax
    struct { int dim; int stride; } softmax_params = {kv_len, (int)max_seq_len};
    MetalManager::dispatch1d(
        "softmax",
        n_heads, 1024,
        &softmax_params, sizeof(softmax_params),
        (MTL::Buffer*[]){att_scores_buf, att_scores_buf}, 2
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
        (MTL::Buffer*[]){att_scores_buf, v_cache, att_out_buf}, 3
    );

    struct { size_t offset; int d_in; } output_args = { 0, q_dim };

    // 13. output proj
    MetalManager::dispatch1d(
        lin_proj_name.c_str(),
        d_model, 32,
        &output_args, sizeof(output_args),
        (MTL::Buffer*[]){wo->metal_buf(), att_out_buf, x_norm}, 3
    );

    // 14. residual
    size_t res_thrgps = (d_model+1023) / 1024;
    MetalManager::dispatch1d(
        "resadd",
        res_thrgps, 1024,
        nullptr, 0,
        (MTL::Buffer*[]){x_in, x_norm}, 2
    );
}

// === MOE ===

Qwen3MoE::Qwen3MoE(
    int d_model, int d_ff, int n_experts, int n_active_experts, 
    float eps,
    TPtr w_moenorm, TPtr w_router,
    TPtr ws_gate, TPtr ws_down, TPtr ws_up,
    DeviceType device
) : MoE(
        d_model, d_ff, n_experts, n_active_experts, eps, 
        w_moenorm, w_router,
        ws_gate, ws_down, ws_up,
        device
    ) {}

void Qwen3MoE::forward(std::shared_ptr<RunState> run_state) {
    if (get_device() == DeviceType::CPU) {
        Qwen3MoE::cpu_forward(
            std::get<float*>(run_state->x), std::get<float*>(run_state->xb), std::get<float*>(run_state->xb2),
            std::get<float*>(run_state->hb), std::get<float*>(run_state->hb2),
            std::get<int*>(run_state->active_experts), std::get<float*>(run_state->active_experts_weights), 
            std::get<float*>(run_state->moe_scores)
        );
    } else if (get_device() == DeviceType::METAL) {
        Qwen3MoE::metal_forward(
            std::get<MTL::Buffer*>(run_state->x), std::get<MTL::Buffer*>(run_state->xb), std::get<MTL::Buffer*>(run_state->xb2),
            std::get<MTL::Buffer*>(run_state->hb), std::get<MTL::Buffer*>(run_state->hb2),
            std::get<MTL::Buffer*>(run_state->active_experts), std::get<MTL::Buffer*>(run_state->active_experts_weights), 
            std::get<MTL::Buffer*>(run_state->moe_scores)
        );
    } else {
        assert(false && "Qwen3MoE support for this device not implemented yet");
    }
}

void Qwen3MoE::cpu_forward(
    float* x_in, float* x_norm,
    float* exp_buf, float* gate_buf, float* up_buf,
    int* active_experts, float* active_experts_weights,
    float* moe_scores
) {
    rmsnorm(x_norm, x_in, w_moenorm->cpu_typed_view<DataType::F32>(), d_model, eps);

    if (n_experts == 0) {
        active_experts[0] = 0;
        active_experts_weights[0] = 1.0f;
    } else {
        route(
            x_norm, active_experts, active_experts_weights,
            moe_scores, w_router->cpu_typed_view<DataType::F32>(),
            d_model, n_experts, n_active_experts
        );
    }

    int n = (n_experts > 0 ? n_active_experts : 1);
    for (int i=0; i < n; ++i) {
        int expert_idx = active_experts[i];

        // SwiGLU
        matmul(gate_buf, x_norm, ws_gate, expert_idx*d_ff*d_model, d_ff, d_model);
        matmul(up_buf, x_norm, ws_up, expert_idx*d_ff*d_model, d_ff, d_model);
        for (int i=0; i<d_ff; ++i) {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }
        matmul(exp_buf, gate_buf, ws_down, expert_idx*d_model*d_ff, d_model, d_ff);

        // resadd
        for (int j=0; j<d_model; ++j) {
            x_in[j] += exp_buf[j]*active_experts_weights[i];
        }
    }
}

void Qwen3MoE::metal_forward(
    MTL::Buffer* x_in, MTL::Buffer* x_norm,
    MTL::Buffer* exp_buf, MTL::Buffer* gate_buf, MTL::Buffer* up_buf,
    MTL::Buffer* active_experts, MTL::Buffer* active_experts_weights, 
    MTL::Buffer* moe_scores
) {

    // should assert matching dtypes of ws_up, ws_gate, ws_down tensors in init.
    std::string lin_proj_name = "linear_proj" + dtype_kernel_suffix(ws_up->dtype);

    // 1. rmsnorm
    struct { int dim; float eps; int stride; } norm_params = {d_model, eps, 0};
    MetalManager::dispatch1d(
        "rmsnorm",
        1, 1024,
        &norm_params, sizeof(norm_params),
        (MTL::Buffer*[]){w_moenorm->metal_buf(), x_in, x_norm}, 3
    );

    // if MoE model
    if (n_experts > 0) {

        // 2. router
        struct { size_t weight_offset; int d_in; } router_args = { 0, d_model };

        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            n_experts, 32,
            &router_args, sizeof(router_args),
            (MTL::Buffer*[]){w_router->metal_buf(), x_norm, moe_scores}, 3
        );

        // 3. top-k
        struct { int n_experts; int k; } topk_params = {n_experts, n_active_experts};
        MetalManager::dispatch1d(
            "moe_topk",
            1, 32,
            &topk_params, sizeof(topk_params),
            (MTL::Buffer*[]){moe_scores, active_experts, active_experts_weights}, 3
        );

        // 4. softmax
        struct { int dim; int stride; } softmax_params = {n_active_experts, 0};
        MetalManager::dispatch1d(
            "softmax",
            1, 1024,
            &softmax_params, sizeof(softmax_params),
            (MTL::Buffer*[]){active_experts_weights, active_experts_weights}, 2
        );
    }

    int* active_experts_cpu = (int*)MetalManager::cpu_ptr(active_experts);
    float* active_experts_weights_cpu = (float*)MetalManager::cpu_ptr(active_experts_weights);

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
            (MTL::Buffer*[]){ws_gate->metal_buf(), x_norm, gate_buf}, 3
        );

        // up proj
        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            d_ff, 32,
            &up_args, sizeof(up_args),
            (MTL::Buffer*[]){ws_up->metal_buf(), x_norm, up_buf}, 3
        );

        // silu+mul
        size_t silu_thrgps = (d_ff+1023) / 1024;
        MetalManager::dispatch1d(
            "silu_mul",
            silu_thrgps, 1024,
            nullptr, 0,
            (MTL::Buffer*[]){gate_buf, up_buf}, 2
        );

        // down proj
        MetalManager::dispatch1d(
            lin_proj_name.c_str(),
            d_model, 32,
            &down_args, sizeof(down_args),
            (MTL::Buffer*[]){ws_down->metal_buf(), gate_buf, exp_buf}, 3
        );

        // weighted resadd
        float weight = active_experts_weights_cpu[i];
        size_t res_thrgps = (d_model+1023) / 1024;
        MetalManager::dispatch1d(
            "weight_resadd",
            res_thrgps, 1024,
            &weight, sizeof(float),
            (MTL::Buffer*[]){x_in, exp_buf}, 2
        );
    }
}