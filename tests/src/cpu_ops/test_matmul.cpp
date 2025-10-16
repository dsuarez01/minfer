#include "cpu_ops/test_matmul.hpp"
#include "minfer/ops/kernels.hpp"
#include "minfer/base/types.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>

TestMatmul::TestMatmul(const std::string& name) : TestBase(name) {}

void TestMatmul::test_identity() {
    float actual[16] = {0.0f};

    float input[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    // 16x16 identity matrix (256 elements, mostly zeros)
    float weight_arr[256] = {0.0f}; // init. all to zero
    for (int i = 0; i < 16; i++) {
        weight_arr[i * 16 + i] = 1.0f; // set diagonal to 1
    }
    
    auto weight = fp32_t(reinterpret_cast<std::byte*>(weight_arr));

    int size = 16;

    float expected[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    matmul_fp32(actual, input, weight, 0, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Identity matrix mult.");
}

void TestMatmul::test_square() {
    float actual[16] = {0.0f};

    float input[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    // 16x16 matrix with first row all 1s, second row all 2s, etc.
    float weight_arr[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            weight_arr[i * 16 + j] = (i == 2) ? -2.0f : 1.0f; // 3rd row negative
        }
    }
    
    auto weight = fp32_t(reinterpret_cast<std::byte*>(weight_arr));

    int size = 16;

    float expected[16];
    float sum = 136.0f; // sum of input vector (1+2+...+16)
    for (int i = 0; i < 16; i++) {
        expected[i] = (i == 2) ? -2.0f * sum : sum;
    }

    matmul_fp32(actual, input, weight, 0, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Square matrix mult.");
}

void TestMatmul::test_rectangular() {
    float actual[16] = {0.0f};

    float input[32] = {1.0f, 2.0f, -3.0f, -4.0f, 0.5f, 1.5f, -0.5f, 2.5f,
                       0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, -0.3f, -0.4f,
                       1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                       -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -2.0f, -2.0f};
    
    // 16x32 weight matrix
    float weight_arr[512];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            weight_arr[i * 32 + j] = (j % 4 == 0) ? 1.0f : 
                                 (j % 4 == 1) ? -0.5f :
                                 (j % 4 == 2) ? 2.0f : 0.25f;
        }
    }
    
    auto weight = fp32_t(reinterpret_cast<std::byte*>(weight_arr));

    int d_out = 16;
    int d_in = 32;

    // calculate expected manually
    float expected[16];
    for (int i = 0; i < 16; i++) {
        expected[i] = 0.0f;
        for (int j = 0; j < 32; j++) {
            expected[i] += input[j] * weight_arr[i * 32 + j];
        }
    }

    matmul_fp32(actual, input, weight, 0, d_out, d_in);

    assert_arrays_equal(expected, actual, d_out, 1e-6f, "Rectangular matrix mult.");
}

void TestMatmul::test_zero_matrix() {
    float actual[16] = {0.0f};

    float input[32] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                       17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                       25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
    
    float weight_arr[512] = {0.0f}; // all zeros

    auto weight = fp32_t(reinterpret_cast<std::byte*>(weight_arr));

    int d_out = 16;
    int d_in = 32;

    float expected[16] = {0.0f};

    matmul_fp32(actual, input, weight, 0, d_out, d_in);

    assert_arrays_equal(expected, actual, d_out, 1e-6f, "Zero matrix mult.");
}

void TestMatmul::test_zero_vector() {
    float actual[16] = {0.0f};

    float input[16] = {0.0f}; // all zeros
    
    float weight_arr[256];
    for (int i = 0; i < 256; i++) {
        weight_arr[i] = i + 1.0f; // fill w/ sequential values starting at 1.0f
    }
    
    auto weight = fp32_t(reinterpret_cast<std::byte*>(weight_arr));

    int size = 16;

    float expected[16] = {0.0f};

    matmul_fp32(actual, input, weight, 0, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Zero vector mult.");
}

void TestMatmul::test_large_values() {
    float actual[16] = {0.0f};

    float input[16] = {1000.0f, -2000.0f, 500.0f, -1500.0f, 
                       100.0f, 200.0f, -300.0f, 400.0f,
                       50.0f, -75.0f, 25.0f, -125.0f,
                       10.0f, 20.0f, -30.0f, 40.0f};
    
    float weight_arr[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            weight_arr[i * 16 + j] = (j % 2 == 0) ? 0.001f : -0.001f;
        }
    }
    
    auto weight = fp32_t(reinterpret_cast<std::byte*>(weight_arr));

    int d_out = 16;
    int d_in = 16;

    float expected[16];
    for (int i = 0; i < 16; i++) {
        expected[i] = 0.0f;
        for (int j = 0; j < 16; j++) {
            expected[i] += input[j] * weight_arr[i * 16 + j];
        }
    }

    matmul_fp32(actual, input, weight, 0, d_out, d_in);

    assert_arrays_equal(expected, actual, 16, 1e-3f, "Large value mult.");
}

void TestMatmul::test_fp16_matmul() {
    int d_out = 16;
    int d_in = 32;
    
    float input[32];
    for (int i = 0; i < 32; i++) {
        input[i] = 1.0f;
    }
    
    uint16_t weight_fp16_arr[512];
    for (int i = 0; i < 512; i++) {
        weight_fp16_arr[i] = 0x3800; // 0.5 in FP16
    }
    auto weight_fp16 = fp16_t(reinterpret_cast<std::byte*>(weight_fp16_arr));
    
    float result_fp16[16] = {0.0f};
    matmul_fp16(result_fp16, input, weight_fp16, 0, d_out, d_in);
    
    // expected: 32 * 1.0 * 0.5 = 16.0
    for (int i = 0; i < d_out; i++) {
        assert_equal(16.0f, result_fp16[i], 1e-3f, "FP16 matmul result at index " + std::to_string(i));
    }
}

void TestMatmul::test_bf16_matmul() {
    int d_out = 16;
    int d_in = 32;
    
    float input[32];
    for (int i = 0; i < 32; i++) {
        input[i] = 1.0f;
    }
    
    uint16_t weight_bf16_arr[512];
    for (int i = 0; i < 512; i++) {
        weight_bf16_arr[i] = 0x3F00;  // 0.5 in BF16
    }
    auto weight_bf16 = bf16_t(reinterpret_cast<std::byte*>(weight_bf16_arr));
    
    float result_bf16[16] = {0.0f};
    matmul_bf16(result_bf16, input, weight_bf16, 0, d_out, d_in);
    
    // expected: 32 * 1.0 * 0.5 = 16.0
    for (int i = 0; i < d_out; i++) {
        assert_equal(16.0f, result_bf16[i], 1e-3f, "BF16 matmul result at index " + std::to_string(i));
    }
}

// meant to test 3D tensors used in MoE layers
void TestMatmul::test_offset() {
    int n_experts = 4;
    int d_out = 8;
    int d_in = 16;
    int expert_id = 2; // test w/ expert 2
    
    float input[16];
    for (int i = 0; i < 16; i++) {
        input[i] = i + 1.0f; // 1, 2, 3, ..., 16
    }
    
    // FP16: create weight tensor [4, 8, 16] = 512 elems
    uint16_t weight_fp16_arr[512];
    for (int i = 0; i < 512; i++) {
        weight_fp16_arr[i] = ((i / (d_out * d_in)) % 2 == 0) ? 0x3C00 : 0xBC00; // 1.0 or -1.0
    }
    auto weight_fp16 = fp16_t(reinterpret_cast<std::byte*>(weight_fp16_arr));
    
    // BF16: same pattern
    uint16_t weight_bf16_arr[512];
    for (int i = 0; i < 512; i++) {
        weight_bf16_arr[i] = ((i / (d_out * d_in)) % 2 == 0) ? 0x3F80 : 0xBF80; // 1.0 or -1.0
    }
    auto weight_bf16 = bf16_t(reinterpret_cast<std::byte*>(weight_bf16_arr));
    
    // FP32: dequantize from FP16
    float weight_f32_arr[512];
    for (int i = 0; i < n_experts * d_out; i++) {
        weight_fp16.dequantize_row(&weight_f32_arr[i * d_in], i, d_in);
    }
    auto weight_f32 = fp32_t(reinterpret_cast<std::byte*>(weight_f32_arr));
    
    float result_f32[8] = {0.0f};
    float result_fp16[8] = {0.0f};
    float result_bf16[8] = {0.0f};
    
    int offset = expert_id * d_out * d_in;
    
    matmul_fp32(result_f32, input, weight_f32, offset, d_out, d_in);
    matmul_fp16(result_fp16, input, weight_fp16, offset, d_out, d_in);
    matmul_bf16(result_bf16, input, weight_bf16, offset, d_out, d_in);
    
    // the FP32 weights are just dequantized FP16 so the tol is abs
    assert_arrays_equal(result_f32, result_fp16, d_out, 1e-6f, "FP16 vs FP32 offset matmul");
    assert_arrays_equal(result_f32, result_bf16, d_out, 1e-6f, "BF16 vs FP32 offset matmul");
}

void TestMatmul::run_all_tests() {
    test_identity();
    test_square();
    test_rectangular();
    test_zero_matrix();
    test_zero_vector();
    test_large_values();
    test_fp16_matmul();
    test_bf16_matmul();
    test_offset();
}