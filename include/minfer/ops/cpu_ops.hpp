#pragma once

#include <cstdint>

struct float_tag {};
struct fp16_tag {};
struct bf16_tag {};

using fp16_t = uint16_t;
using bf16_t = uint16_t;

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_BF16)
    #include <arm_bf16.h>
#endif

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
inline float half_to_float(fp16_t x) {
    __fp16 half_val = *(__fp16*)&x;
    return (float)half_val;  // Automatic promotion
}
inline fp16_t float_to_half(float x) {
    __fp16 half_val = (__fp16)x;
    return *(fp16_t*)&half_val;
}
#elif defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC)
inline float half_to_float(bf16_t x) {
    bfloat16_t bf16_val = *(bfloat16_t*)&x;
    return vcvtah_f32_bf16(bf16_val);
}
inline bf16_t float_to_half(float x) {
    bfloat16_t bf16_val = vcvth_bf16_f32(x);
    return *(bf16_t*)&bf16_val;
}
#elif defined(__F16C__)
#include <immintrin.h>
inline float half_to_float(fp16_t x) {
    return _cvtsh_ss(x);
}
inline fp16_t float_to_half(float x) {
    return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(uint16_t x) {
    assert(false && "This platform doesn't support FP16 or BF16. Check compiler flags");
    return 0.0f;
}
inline uint16_t float_to_half(float x) {
    assert(false && "This platform doesn't support FP16 or BF16. Check compiler flags");
    return 0;
}
#endif

namespace cpu {

    float silu(float x);
    void softmax(float* x_out, const float* x_in, int size);
    void il_rope(float* x_out, const float* x_in, int d_flat, int d_head, 
            int d_rotary, float freq_base, int pos);
    void neox_rope(float* x_out, const float* x_in, int d_flat, int d_head, 
            int d_rotary, float freq_base, int pos);
    void attn(float* att_scores, float* att_out, const float* q_head, const float* kh, const float* vh,
            int seq_len, int d_head, int k_dim, int v_dim);
    void route(const float* x_norm, int* active_experts, float* active_experts_scores, 
            float* active_experts_weights, float* moe_scores, const float* w_router,
            int d_model, int n_experts, int n_active_experts);

    // Explicit specialization declarations
    template<typename WeightType, typename Tag>
    void matmul(float* x_out, const float* x_in, const WeightType* weight, int d_out, int d_in);
    
    template<>
    void matmul<float, float_tag>(float* x_out, const float* x_in, const float* weight, int d_out, int d_in);
    
    template<>
    void matmul<fp16_t, fp16_tag>(float* x_out, const float* x_in, const fp16_t* weight, int d_out, int d_in);
    
    template<>
    void matmul<bf16_t, bf16_tag>(float* x_out, const float* x_in, const bf16_t* weight, int d_out, int d_in);

    template <typename WeightType, typename Tag>
    void swiglu(const float* x_in, float* exp_buf, float* gate_buf, float* up_buf,
                const WeightType* w_gate, const WeightType* w_up, const WeightType* w_down, 
                int d_ff, int d_model);

    void rmsnorm(float* x_out, const float* x_in, const float* weight, int dim, float eps);
};