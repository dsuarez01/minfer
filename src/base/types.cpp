#include "minfer/base/types.hpp"

#if USE_ARM64
    #include <arm_neon.h>
    #if USE_FP16
        #include <arm_fp16.h>
    #endif
    #if USE_BF16
        #include <arm_bf16.h>
    #endif
#endif

#if USE_ARM64 && USE_FP16
void fp16_t::dequantize_row(float* out, size_t row_idx, size_t d_in) {
    const __fp16* src = reinterpret_cast<const __fp16*>(ptr(row_idx * d_in));
    for (int i=0; i<d_in; ++i) {
        out[i] = static_cast<float>(src[i]);
    }
}
#else
void fp16_t::dequantize_row(float*, size_t, size_t) {
    assert(false && "FP16 dequantize_row: USE_ARM64 or USE_FP16 not set/defined")
}
#endif

#if USE_ARM64 && USE_BF16
void bf16_t::dequantize_row(float* out, size_t row_idx, size_t d_in) {
    const __bf16* src = reinterpret_cast<const __bf16*>(ptr(row_idx * d_in));
    for (int i=0; i<d_in; ++i) {
        out[i] = vcvtah_f32_bf16(src[i]);
    }
}
#else
void bf16_t::dequantize_row(float*, size_t, size_t) {
    assert(false && "FP16 dequantize_row: USE_ARM64 or USE_FP16 not set/defined")
}
#endif