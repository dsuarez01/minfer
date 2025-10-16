#include "minfer/ops/kernels.hpp"
#include "minfer/base/tensor.hpp"
#include "minfer/base/types.hpp"
#include "minfer/base/config.hpp"

#include <cmath>
#include <cassert>
#include <cfloat>
#include <bitset>
#include <stdexcept>
#include <variant>
#include <memory>
#include <iostream>

#if USE_ARM64
    #include <arm_neon.h>
    #if USE_FP16
        #include <arm_fp16.h>
    #endif
    #if USE_BF16
        #include <arm_bf16.h>
    #endif
#endif

static constexpr int MAX_EXPERTS = 256; // adjust as needed, for visited set in router

float silu(const float x) {
    return x / (1.0f+std::expf(-x));
}

void softmax(float* x_out, const float* x_in, int size) {
    float max_val = -FLT_MAX;
    for (int i=0; i<size; ++i) {
        if (x_in[i] > max_val) {
            max_val = x_in[i];
        }
    }
    
    float sum_centered = 0.0f;
    for (int i=0; i<size; ++i) {
        x_out[i] = std::expf(x_in[i] - max_val);
        sum_centered += x_out[i];
    }

    float inv_sum = 1.0f / sum_centered;
    for (int i=0; i<size; ++i) {
        x_out[i] *= inv_sum;
    }
}

void il_rope(float* x_out, const float* x_in, int head_idx, int d_head, int d_rotary, float freq_base, int pos) {
    const float* head_in = x_in + head_idx*d_head;
    float* head_out = x_out + head_idx*d_head;

    for (int pair_idx=0; pair_idx<d_rotary/2; ++pair_idx) {
        float freq = 1.0f / pow(freq_base, 2.0f*pair_idx/d_rotary);
        float angle = pos*freq;
        
        float x_0 = head_in[2*pair_idx];
        float x_1 = head_in[2*pair_idx+1];

        head_out[2*pair_idx] = cos(angle)*x_0 - sin(angle)*x_1;
        head_out[2*pair_idx+1] = sin(angle)*x_0 + cos(angle)*x_1;
    }
}

void neox_rope(float* x_out, const float* x_in, int head_idx, int d_head, int d_rotary, float freq_base, int pos) {
    const float* head_in = x_in + head_idx*d_head;
    float* head_out = x_out + head_idx*d_head;

    for (int pair_idx=0; pair_idx<d_rotary/2; ++pair_idx) {
        float freq = 1.0f / pow(freq_base, 2.0f*pair_idx/d_rotary);
        float angle = pos*freq;
        
        float x_0 = head_in[pair_idx];
        float x_1 = head_in[pair_idx+d_rotary/2];

        head_out[pair_idx] = cos(angle)*x_0 - sin(angle)*x_1;
        head_out[pair_idx+d_rotary/2] = sin(angle)*x_0 + cos(angle)*x_1;
    }
}

void attn(
    float* att_scores, float* att_out, 
    const float* q_head, const float* kh, const float* vh,
    int seq_len, int d_head, int kv_dim
) {
    float scale = 1.0f / std::sqrtf((float)d_head);
    
    for (int pos=0; pos<seq_len; ++pos) {
        float score = 0.0f;
        for (int d=0; d<d_head; ++d) {
            score += q_head[d] * kh[pos*kv_dim+d];
        }
        att_scores[pos] = score * scale;
    }
    
    softmax(att_scores, att_scores, seq_len);

    for (int d=0; d<d_head; ++d) {
        att_out[d] = 0.0f;
    }

    for (int pos=0; pos < seq_len; ++pos) {
        float score = att_scores[pos];
        for (int d=0; d<d_head; ++d) {
            att_out[d] += score * vh[pos*kv_dim+d];
        }
    }
}

// assume w_router always FP32
void route(
    const float* x_norm, int* active_experts, float* active_experts_weights, 
    float* moe_scores, const fp32_t& w_router,
    int d_model, int n_experts, int n_active_experts
) {
    matmul_fp32(moe_scores, x_norm, w_router, 0, n_experts, d_model);
    
    std::bitset<MAX_EXPERTS> visited;
    
    for (int i=0; i<n_active_experts; ++i) {
        int best = 0;
        float best_score = -FLT_MAX;
        
        for (int j=0; j<n_experts; ++j) {
            if (!visited[j] && moe_scores[j] > best_score) {
                best = j;
                best_score = moe_scores[j];
            }
        }
        
        active_experts[i] = best;
        active_experts_weights[i] = best_score;
        visited[best] = true;
    }

    softmax(active_experts_weights, active_experts_weights, n_active_experts);
}

// void matmul(float* x_out, const float* x_in, const fp32_t* weight, int d_out, int d_in) {
//     const float* w = weight.ptr();
//     #pragma omp parallel for
//     for (int i=0; i<d_out; i++) {
//         float cur_sum = 0.0f;
//         for (int j=0; j<d_in; j++) {
//             cur_sum += w[i*d_in+j] * x_in[j];
//         }
//         x_out[i] = cur_sum;
//     }
// }

void embed(float* x_out, const TPtr weight, uint32_t token_id, size_t d_in) {
    switch (weight->dtype) {
        case DataType::F32: {
            auto& typed_weight = weight->cpu_typed_view<DataType::F32>();
            embed_fp32(x_out, typed_weight, token_id, d_in);
            break;
        }
        case DataType::F16: {
            auto& typed_weight = weight->cpu_typed_view<DataType::F16>();
            embed_fp16(x_out, typed_weight, token_id, d_in);
            break;
        }
        case DataType::BF16: {
            auto& typed_weight = weight->cpu_typed_view<DataType::BF16>();
            embed_bf16(x_out, typed_weight, token_id, d_in);
            break;
        }
        default: throw std::runtime_error("Unhandled DataType"); break;
    }
}

void embed_fp32(float* x_out, const fp32_t& weight, uint32_t token_id, size_t d_in) {
    const float* w = weight.ptr(token_id * d_in);
    for (size_t i=0; i<d_in; ++i) {
        x_out[i] = w[i];
    }
}

void embed_fp16(float* x_out, const fp16_t& weight, uint32_t token_id, size_t d_in) {
    weight.dequantize_row(x_out, token_id, d_in);
}

void embed_bf16(float* x_out, const bf16_t& weight, uint32_t token_id, size_t d_in) {
    weight.dequantize_row(x_out, token_id, d_in);
}


void matmul(float* x_out, const float* x_in, const TPtr weight, size_t offset, int d_out, int d_in) {
    switch (weight->dtype) {
        case DataType::F32: {
            auto& typed_weight = weight->cpu_typed_view<DataType::F32>();
            matmul_fp32(x_out, x_in, typed_weight, offset, d_out, d_in);
            break;
        }
        case DataType::F16: {
            auto& typed_weight = weight->cpu_typed_view<DataType::F16>();
            matmul_fp16(x_out, x_in, typed_weight, offset, d_out, d_in);
            break;
        }
        case DataType::BF16: {
            auto& typed_weight = weight->cpu_typed_view<DataType::BF16>();
            matmul_bf16(x_out, x_in, typed_weight, offset, d_out, d_in);
            break;
        }
        default: throw std::runtime_error("Unhandled DataType"); break;
    }
}

#if USE_ARM64
void matmul_fp32(float* x_out, const float* x_in, const fp32_t& weight, size_t offset, int d_out, int d_in) {
    const int TILE_SIZE=16;
    assert((d_in % TILE_SIZE == 0) && "d_in should be divisible by the tile size");

    const float* w = weight.ptr(offset);

    #pragma omp parallel for
    for (int i=0; i<d_out; i+=TILE_SIZE) {
        int i_end = std::min(i+TILE_SIZE, d_out);
        for (int ii=i; ii<i_end; ++ii) {
            
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            
            for (int j=0; j<d_in; j+=TILE_SIZE) {
                for (int jj=j; jj<j+TILE_SIZE; jj+=16) {
                    int base_offset = ii*d_in + jj;

                    acc0 = vfmaq_f32(acc0, vld1q_f32(x_in+jj+0),  vld1q_f32(w+base_offset+0));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(x_in+jj+4),  vld1q_f32(w+base_offset+4));
                    acc2 = vfmaq_f32(acc2, vld1q_f32(x_in+jj+8),  vld1q_f32(w+base_offset+8));
                    acc3 = vfmaq_f32(acc3, vld1q_f32(x_in+jj+12), vld1q_f32(w+base_offset+12));
                }
            }

            x_out[ii] = vaddvq_f32(vaddq_f32(acc0, acc1) + vaddq_f32(acc2, acc3));
        }
    }
}
#else
void matmul_fp32(float*, const float*, const fp32_t&, size_t, int, int) {
    assert(false && "Matmul FP32: USE_ARM64 not enabled");
}
#endif

#if USE_ARM64 && USE_FP16
void matmul_fp16(float* x_out, const float* x_in, const fp16_t& weight, size_t offset, int d_out, int d_in) {
    const int TILE_SIZE = 16;
    assert((d_in % TILE_SIZE == 0) && "d_in should be divisible by the tile size");

    const __fp16* w = reinterpret_cast<const __fp16*>(weight.ptr(offset));

    #pragma omp parallel for
    for (int i=0; i<d_out; i+=TILE_SIZE) {
        int i_end = std::min(i+TILE_SIZE, d_out);
        
        for (int ii = i; ii < i_end; ++ii) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            
            for (int j=0; j<d_in; j+=TILE_SIZE) {
                for (int jj = j; jj<j+TILE_SIZE; jj+=16) {
                    int base_offset = ii*d_in + jj;
                    
                    float16x8_t w01 = vld1q_f16(w+base_offset+0);
                    float32x4_t w0 = vcvt_f32_f16(vget_low_f16(w01));
                    float32x4_t w1 = vcvt_f32_f16(vget_high_f16(w01));
                    float16x8_t w23 = vld1q_f16(w+base_offset+8);
                    float32x4_t w2 = vcvt_f32_f16(vget_low_f16(w23));
                    float32x4_t w3 = vcvt_f32_f16(vget_high_f16(w23));

                    float32x4_t x0 = vld1q_f32(x_in+jj+0);
                    float32x4_t x1 = vld1q_f32(x_in+jj+4);
                    float32x4_t x2 = vld1q_f32(x_in+jj+8);
                    float32x4_t x3 = vld1q_f32(x_in+jj+12);

                    acc0 = vfmaq_f32(acc0, x0, w0);
                    acc1 = vfmaq_f32(acc1, x1, w1);
                    acc2 = vfmaq_f32(acc2, x2, w2);
                    acc3 = vfmaq_f32(acc3, x3, w3);
                }
            }
            
            x_out[ii] = vaddvq_f32(vaddq_f32(acc0, acc1) + vaddq_f32(acc2, acc3));
        }
    }
}
#else
void matmul_fp16(float*, const float*, const fp16_t&, size_t, int, int) {
    assert(false && "Matmul FP16: USE_ARM64 or USE_FP16 not enabled");
}
#endif


#if USE_ARM64 && USE_BF16 
void matmul_bf16(float* x_out, const float* x_in, const bf16_t& weight, size_t offset, int d_out, int d_in) {
    const int TILE_SIZE = 16;
    assert((d_in % TILE_SIZE == 0) && "d_in should be divisible by the tile size");

    const __bf16* w = reinterpret_cast<const __bf16*>(weight.ptr(offset));

    #pragma omp parallel for
    for (int i=0; i<d_out; i+=TILE_SIZE) {
        int i_end = std::min(i+TILE_SIZE, d_out);
        
        for (int ii = i; ii < i_end; ++ii) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            
            for (int j=0; j<d_in; j+=TILE_SIZE) {
                for (int jj = j; jj<j+TILE_SIZE; jj+=16) {
                    int base_offset = ii*d_in + jj;

                    bfloat16x8_t w01 = vld1q_bf16(w+base_offset+0);
                    float32x4_t w0 = vcvt_f32_bf16(vget_low_bf16(w01));
                    float32x4_t w1 = vcvt_f32_bf16(vget_high_bf16(w01));
                    bfloat16x8_t w23 = vld1q_bf16(w+base_offset+8);
                    float32x4_t w2 = vcvt_f32_bf16(vget_low_bf16(w23));
                    float32x4_t w3 = vcvt_f32_bf16(vget_high_bf16(w23));

                    float32x4_t x0 = vld1q_f32(x_in+jj+0);
                    float32x4_t x1 = vld1q_f32(x_in+jj+4);
                    float32x4_t x2 = vld1q_f32(x_in+jj+8);
                    float32x4_t x3 = vld1q_f32(x_in+jj+12);

                    acc0 = vfmaq_f32(acc0, x0, w0);
                    acc1 = vfmaq_f32(acc1, x1, w1);
                    acc2 = vfmaq_f32(acc2, x2, w2);
                    acc3 = vfmaq_f32(acc3, x3, w3);
                }
            }
            
            x_out[ii] = vaddvq_f32(vaddq_f32(acc0, acc1) + vaddq_f32(acc2, acc3));
        }
    }
}
#else
void matmul_bf16(float*, const float*, const bf16_t&, size_t, int, int) {
    assert(false && "Matmul BF16: USE_ARM64 or USE_BF16 not enabled");
}
#endif


void rmsnorm(float* x_out, const float* x_in, const fp32_t& weight, int dim, float eps) {
    
    const float* w = weight.ptr();

    float rms = 0.0f;
    for (int i=0; i<dim; ++i) {
        rms += x_in[i]*x_in[i];
    }
    rms = sqrtf(rms/dim + eps);
    float inv_rms = 1.0f/rms;

    for (int i=0; i<dim; ++i) {
        x_out[i] = inv_rms*x_in[i]*w[i];
    }
}