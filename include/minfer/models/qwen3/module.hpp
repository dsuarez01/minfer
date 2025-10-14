#pragma once

#include "minfer/base/config.hpp"
#include "minfer/base/module.hpp"

namespace MTL {
    class Buffer;
}

class Qwen3Embed : public Embed {
public:
    Qwen3Embed(
        size_t vocab_size, int d_model, 
        TPtr weight,
        DeviceType device = DeviceType::CPU
    );

    void forward(std::shared_ptr<RunState> run_state) override;

private:
    void cpu_forward(float* x_out, int token_id);
    void metal_forward(MTL::Buffer* x_out, int token_id);
};

class Qwen3LMHead : public Linear {
public:
    Qwen3LMHead(
        int d_in, int d_out,
        TPtr weight, TPtr bias,
        DeviceType device = DeviceType::CPU
    );
    void forward(std::shared_ptr<RunState> run_state) override;

private:
    void cpu_forward(float* x_out, float* x_in);
    void metal_forward(MTL::Buffer* x_out, MTL::Buffer* x_in);
};

class Qwen3FinalRMSNorm : public RMSNorm {
public:
    Qwen3FinalRMSNorm(
        int dim, float eps, 
        TPtr weight,
        DeviceType device = DeviceType::CPU
    );
    void forward(std::shared_ptr<RunState> run_state) override;

private:
    void cpu_forward(float* x_out, float* x_in);
    void metal_forward(MTL::Buffer* x_out, MTL::Buffer* x_in);
};

class Qwen3GQA : public GQA {
public:
    Qwen3GQA(
        int block_idx, int d_model, size_t max_seq_len,
        int n_heads, int n_kv_heads, int d_head, int d_rotary,
        float eps, float freq_base,
        TPtr wq, TPtr wk, TPtr wv,
        TPtr wo, TPtr wq_norm, TPtr wk_norm,
        TPtr w_attnnorm, 
        DeviceType device = DeviceType::CPU
    );
    void forward(std::shared_ptr<RunState> run_state) override;

private:
    void cpu_forward(
        float* x_in, float* x_norm, 
        float* att_out_buf, float* att_scores_buf, 
        float* q_buf, float* k_buf, float* v_buf, 
        float* k_cache, float* v_cache,
        int cur_pos
    );
    void metal_forward(
        MTL::Buffer* x_in, MTL::Buffer* x_norm, 
        MTL::Buffer* att_out_buf, MTL::Buffer* att_scores_buf, 
        MTL::Buffer* q_buf, MTL::Buffer* k_buf, MTL::Buffer* v_buf, 
        MTL::Buffer* k_cache, MTL::Buffer* v_cache,
        int cur_pos
    );
};

class Qwen3MoE : public MoE {
public:
    Qwen3MoE(
        int d_model, int d_ff, int n_experts, int n_active_experts, float eps,
        TPtr w_moenorm, TPtr w_router,
        TPtr ws_gate, TPtr ws_down, TPtr ws_up,
        DeviceType device = DeviceType::CPU
    );
    void forward(std::shared_ptr<RunState> run_state) override;

private:
    void cpu_forward(
        float* x_in, float* x_norm,
        float* exp_buf, float* gate_buf, float* up_buf,
        int* active_experts, float* active_experts_weights, 
        float* moe_scores
    );

    void metal_forward(
        MTL::Buffer* x_in, MTL::Buffer* x_norm,
        MTL::Buffer* exp_buf, MTL::Buffer* gate_buf, MTL::Buffer* up_buf,
        MTL::Buffer* active_experts, MTL::Buffer* active_experts_weights, 
        MTL::Buffer* moe_scores
    );
};