#include "minfer/ops/cpu_ops.hpp"
#include "minfer/base/module.hpp"

#include <cmath>
#include <cfloat>
#include <bitset>
#include <Accelerate/Accelerate.h>

namespace cpu {
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

    void il_rope(float* x_out, const float* x_in, int d_flat, int d_head, 
        int d_rotary, float freq_base, int pos, std::vector<float>& rope_table) {
        for (int i=0; i<d_flat; i+=2) {
            float x_0 = x_in[i];
            float x_1 = x_in[i+1];

            int ii = i % d_head;
            // float freq = ii >= d_rotary ? 0.0f : 1.0f / std::powf(freq_base, (float) ii/ (float) d_rotary);
            // float angle = pos*freq;
            
            // float c_angle = std::cosf(angle);
            // float s_angle = std::sinf(angle);

            float cos_angle = ii >= d_rotary ? 1.0f : rope_table[pos*d_rotary + 2*ii];
            float sine_angle = ii >= d_rotary ? 0.0f : rope_table[pos*d_rotary + 2*ii+1];

            float new_x_0 = cos_angle*x_0 - sine_angle*x_1;
            float new_x_1 = sine_angle*x_0 + cos_angle*x_1;
            
            x_out[i] = new_x_0;
            x_out[i+1] = new_x_1;
        }
    }

    void neox_rope(
        float* x_out, const float* x_in, int d_flat, int d_head, 
        int d_rotary, float freq_base, int pos, std::vector<float>& rope_table
    ) {
        for (int i=0; i<d_flat; ) {
            int i_0 = i%d_head;
            int i_1 = i_0 < d_rotary/2 ? i_0 + d_rotary/2 : i_0;
            float x_0 = x_in[i_0 + (i / d_head) * d_head];
            float x_1 = x_in[i_1 + (i / d_head) * d_head]; 

            // float freq = i_0 < d_rotary / 2 ? 1.0f / std::powf(freq_base, 2.0f * i_0 / d_rotary) : 0.0f;
            // float angle = pos * freq;
            float cos_angle = i_0 >= d_rotary / 2 ? 1.0f : rope_table[pos*d_rotary + 2*i_0];
            float sine_angle = i_0 >= d_rotary / 2 ? 0.0f : rope_table[pos*d_rotary + 2*i_0+1];

            x_out[i_0 + (i / d_head) * d_head] = cos_angle * x_0 - sine_angle * x_1;
            x_out[i_1 + (i / d_head) * d_head] = sine_angle * x_0 + cos_angle * x_1;
            i += ((i+1)%d_head == d_rotary/2) ? d_rotary/2 + 1 : 1;
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
        
        // for (int pos = 0; pos<seq_len; ++pos) {
        //     float score = att_scores[pos];
        //     for (int d = 0; d<d_head; ++d) {
        //         att_out[d] = (pos == 0) ? 
        //             score*vh[pos * kv_dim + d] :
        //             att_out[d]+score*vh[pos*kv_dim + d];
        //     }
        // }

        // for (int d=0; d<d_head; ++d) {
        //     float output = 0.0f;
        //     for (int pos=0; pos < seq_len; ++pos) {
        //         output += att_scores[pos] * vh[pos*kv_dim+d];
        //     }
        //     att_out[d] = output;
        // }

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
        const float* x_norm, int* active_experts, float* active_experts_scores, 
        float* active_experts_weights, float* moe_scores, const float* w_router,
        int d_model, int n_experts, int n_active_experts
    ) {
        matmul<float, float_tag>(moe_scores, x_norm, w_router, n_experts, d_model);
        
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
            active_experts_scores[i] = best_score;
            visited[best] = true;
        }

        softmax(active_experts_weights, active_experts_scores, n_active_experts);
    }

    template<typename WeightType, typename Tag> // should never be called
    void matmul(float* x_out, const float* x_in, const WeightType* weight, int d_out, int d_in) {
        static_assert(sizeof(WeightType) == 0, "Matmul: unsupported WeightType/Tag");
    }

    // template<>
    // void matmul<float, float_tag>(float* x_out, const float* x_in, const float* weight, int d_out, int d_in) {
    //     #pragma omp parallel for
    //     for (int i=0; i<d_out; i++) {
    //         float cur_sum = 0.0f;
    //         for (int j=0; j<d_in; j++) {
    //             cur_sum += weight[i*d_in+j] * x_in[j];
    //         }
    //         x_out[i] = cur_sum;
    //     }
    // }

    template<>
    void matmul<float, float_tag>(float* x_out, const float* x_in, const float* weight, int d_out, int d_in) {
        
        const int TILE_SIZE=16; // needs to be a multiple of 16 (4 Neon pipelines)

        assert((d_in % TILE_SIZE == 0) && "d_in should both be divisible by the tile size");

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

                        acc0 = vfmaq_f32(acc0, vld1q_f32(x_in+jj+0),  vld1q_f32(weight+base_offset+0));
                        acc1 = vfmaq_f32(acc1, vld1q_f32(x_in+jj+4),  vld1q_f32(weight+base_offset+4));
                        acc2 = vfmaq_f32(acc2, vld1q_f32(x_in+jj+8),  vld1q_f32(weight+base_offset+8));
                        acc3 = vfmaq_f32(acc3, vld1q_f32(x_in+jj+12), vld1q_f32(weight+base_offset+12));
                    }
                    
                }

                x_out[ii] = vaddvq_f32(vaddq_f32(acc0, acc1) + vaddq_f32(acc2, acc3));
            }
        }
    }

    template<>
    void matmul<fp16_t, fp16_tag>(float* x_out, const float* x_in, const fp16_t* weight, int d_out, int d_in) {
        const int TILE_SIZE = 16; // needs to be a multiple of 16 (4 Neon pipelines)
        
        assert((d_in % TILE_SIZE == 0) && "d_in should both be divisible by the tile size");

        #pragma omp parallel for
        for (int i=0; i<d_out; i+=TILE_SIZE) {
            int i_end = std::min(i+TILE_SIZE, d_out);
            
            for (int ii = i; ii < i_end; ++ii) { // rows w/in tile processed one at a time
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);
                
                
                for (int j=0; j<d_in; j+=TILE_SIZE) {
                    
                    for (int jj = j; jj<j+TILE_SIZE; jj+=16) { // columns w/in tile processed 16 a time (4 Neon pipelines)
                        int base_offset = ii*d_in + jj;
                        
                        float16x8_t w01 = vld1q_f16((const __fp16*)(weight+base_offset+0));
                        float32x4_t w0 = vcvt_f32_f16(vget_low_f16(w01));
                        float32x4_t w1 = vcvt_f32_f16(vget_high_f16(w01));
                        float16x8_t w23 = vld1q_f16((const __fp16*)(weight+base_offset+8));
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

    template<>
    void matmul<bf16_t, bf16_tag>(float* x_out, const float* x_in, const bf16_t* weight, int d_out, int d_in) {
        const int TILE_SIZE = 16; // needs to be a multiple of 16 (4 Neon pipelines)
        
        assert((d_in % TILE_SIZE == 0) && "d_in should both be divisible by the tile size");

        #pragma omp parallel for
        for (int i=0; i<d_out; i+=TILE_SIZE) {
            int i_end = std::min(i+TILE_SIZE, d_out);
            
            for (int ii = i; ii < i_end; ++ii) { // rows w/in tile processed one at a time
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);
                
                
                for (int j=0; j<d_in; j+=TILE_SIZE) {
                    
                    for (int jj = j; jj<j+TILE_SIZE; jj+=16) { // columns w/in tile processed 16 a time (4 Neon pipelines)
                        int base_offset = ii*d_in + jj;

                        bfloat16x8_t w01 = vld1q_bf16((const __bf16*)(weight+base_offset+0));
                        float32x4_t w0 = vcvt_f32_bf16(vget_low_bf16(w01));
                        float32x4_t w1 = vcvt_f32_bf16(vget_high_bf16(w01));
                        bfloat16x8_t w23 = vld1q_bf16((const __bf16*)(weight+base_offset+8));
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

    void rmsnorm(float* x_out, const float* x_in, const float* weight, int dim, float eps) {
        float rms = 0.0f;
        for (int i=0; i<dim; ++i) {
            rms += x_in[i]*x_in[i];
        }
        rms = sqrtf(rms/dim + eps);
        float inv_rms = 1.0f/rms;

        for (int i=0; i<dim; ++i) {
            x_out[i] = inv_rms*x_in[i]*weight[i];
        }
    }

    template <typename WeightType, typename Tag>
    void swiglu(const float* x_in, float* exp_buf, float* gate_buf, float* up_buf,
                const WeightType* w_gate, const WeightType* w_up, const WeightType* w_down, 
                int d_ff, int d_model) {
        
        matmul<WeightType, Tag>(gate_buf, x_in, w_gate, d_ff, d_model);
        matmul<WeightType, Tag>(up_buf, x_in, w_up, d_ff, d_model);
        
        for (int i=0; i<d_ff; ++i) {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }

        matmul<WeightType, Tag>(exp_buf, gate_buf, w_down, d_model, d_ff);
    }

    // explicit instantiations
    template void cpu::swiglu<float, float_tag>(const float*, float*, float*, float*, const float*, const float*, const float*, int, int);
    template void cpu::swiglu<fp16_t, fp16_tag>(const float*, float*, float*, float*, const fp16_t*, const fp16_t*, const fp16_t*, int, int);
    template void cpu::swiglu<bf16_t, bf16_tag>(const float*, float*, float*, float*, const bf16_t*, const bf16_t*, const bf16_t*, int, int);
}