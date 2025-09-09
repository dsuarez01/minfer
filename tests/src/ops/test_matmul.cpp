#include "ops/test_matmul.hpp"
#include "minfer/ops/cpu_ops.hpp"

TestMatmul::TestMatmul(const std::string& name) : TestBase(name) {}

void TestMatmul::test_identity() {
    float actual[16] = {0.0f};

    float input[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    // 16x16 identity matrix (256 elements, mostly zeros)
    float weight[256] = {0.0f}; // init. all to zero
    for (int i = 0; i < 16; i++) {
        weight[i * 16 + i] = 1.0f; // set diagonal to 1
    }
    
    int size = 16;

    float expected[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Identity matrix mult.");
}

void TestMatmul::test_square() {
    float actual[16] = {0.0f};

    float input[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    // 16x16 matrix with first row all 1s, second row all 2s, etc.
    float weight[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            weight[i * 16 + j] = (i == 2) ? -2.0f : 1.0f; // 3rd row negative
        }
    }
    
    int size = 16;

    float expected[16];
    float sum = 136.0f; // sum of input vector (1+2+...+16)
    for (int i = 0; i < 16; i++) {
        expected[i] = (i == 2) ? -2.0f * sum : sum;
    }

    cpu::matmul<float, float_tag>(actual, input, weight, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Square matrix mult.");
}

void TestMatmul::test_rectangular() {
    float actual[16] = {0.0f};

    float input[32] = {1.0f, 2.0f, -3.0f, -4.0f, 0.5f, 1.5f, -0.5f, 2.5f,
                       0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, -0.3f, -0.4f,
                       1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                       -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -2.0f, -2.0f};
    
    // 16x32 weight matrix
    float weight[512];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            weight[i * 32 + j] = (j % 4 == 0) ? 1.0f : 
                                 (j % 4 == 1) ? -0.5f :
                                 (j % 4 == 2) ? 2.0f : 0.25f;
        }
    }
    
    int d_out = 16;
    int d_in = 32;

    // calculate expected manually
    float expected[16];
    for (int i = 0; i < 16; i++) {
        expected[i] = 0.0f;
        for (int j = 0; j < 32; j++) {
            expected[i] += input[j] * weight[i * 32 + j];
        }
    }

    cpu::matmul<float, float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, d_out, 1e-6f, "Rectangular matrix mult.");
}

void TestMatmul::test_zero_matrix() {
    float actual[16] = {0.0f};

    float input[32] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                       17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                       25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
    
    float weight[512] = {0.0f}; // all zeros
    int d_out = 16;
    int d_in = 32;

    float expected[16] = {0.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, d_out, 1e-6f, "Zero matrix mult.");
}

void TestMatmul::test_zero_vector() {
    float actual[16] = {0.0f};

    float input[16] = {0.0f}; // all zeros
    
    float weight[256];
    for (int i = 0; i < 256; i++) {
        weight[i] = i + 1.0f; // fill w/ sequential values starting at 1.0f
    }
    
    int size = 16;

    float expected[16] = {0.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Zero vector mult.");
}

void TestMatmul::test_large_values() {
    float actual[16] = {0.0f};

    float input[16] = {1000.0f, -2000.0f, 500.0f, -1500.0f, 
                       100.0f, 200.0f, -300.0f, 400.0f,
                       50.0f, -75.0f, 25.0f, -125.0f,
                       10.0f, 20.0f, -30.0f, 40.0f};
    
    float weight[256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            weight[i * 16 + j] = (j % 2 == 0) ? 0.001f : -0.001f;
        }
    }
    
    int d_out = 16;
    int d_in = 16;

    float expected[16];
    for (int i = 0; i < 16; i++) {
        expected[i] = 0.0f;
        for (int j = 0; j < 16; j++) {
            expected[i] += input[j] * weight[i * 16 + j];
        }
    }

    cpu::matmul<float, float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, 16, 1e-3f, "Large value mult.");
}

void TestMatmul::test_fp16_matmul() {
    const int d_out = 16;
    const int d_in = 32;
    
    float input[32];
    for (int i = 0; i < 32; i++) {
        input[i] = (i % 4 == 0) ? 1.0f : 
                   (i % 4 == 1) ? 2.0f :
                   (i % 4 == 2) ? -0.5f : 1.5f;
    }
    
    float weight_f32[512];
    fp16_t weight_fp16[512];
    
    for (int i = 0; i < 512; i++) {
        weight_f32[i] = (i % 8 < 4) ? 1.0f : -0.5f;
        weight_fp16[i] = float_to_half(weight_f32[i]);
    }
    
    float result_f32[16] = {0.0f};
    float result_fp16[16] = {0.0f};
    
    cpu::matmul<float, float_tag>(result_f32, input, weight_f32, d_out, d_in);
    cpu::matmul<fp16_t, fp16_tag>(result_fp16, input, weight_fp16, d_out, d_in);
    
    for (int i = 0; i < d_out; i++) {
        float tolerance = std::max(1e-2f, std::abs(result_f32[i]) * 1e-2f);
        assert_equal(result_f32[i], result_fp16[i], tolerance,
                    "FP16 vs F32 matmul result at index " + std::to_string(i));
    }
}

void TestMatmul::test_bf16_matmul() {
    const int d_out = 16;
    const int d_in = 32;
    
    float input[32];
    for (int i = 0; i < 32; i++) {
        input[i] = (i % 4 == 0) ? 1.0f : 
                   (i % 4 == 1) ? 2.0f :
                   (i % 4 == 2) ? -0.5f : 1.5f;
    }
    
    float weight_f32[512];
    bf16_t weight_bf16[512];
    
    for (int i = 0; i < 512; i++) {
        weight_f32[i] = (i % 8 < 4) ? 1.0f : -0.5f;
        weight_bf16[i] = float_to_half(weight_f32[i]);
    }
    
    float result_f32[16] = {0.0f};
    float result_bf16[16] = {0.0f};
    
    cpu::matmul<float, float_tag>(result_f32, input, weight_f32, d_out, d_in);
    cpu::matmul<bf16_t, bf16_tag>(result_bf16, input, weight_bf16, d_out, d_in);
    
    for (int i = 0; i < d_out; i++) {
        float tolerance = std::max(1e-2f, std::abs(result_f32[i]) * 1e-2f);
        assert_equal(result_f32[i], result_bf16[i], tolerance,
                    "BF16 vs F32 matmul result at index " + std::to_string(i));
    }
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
}