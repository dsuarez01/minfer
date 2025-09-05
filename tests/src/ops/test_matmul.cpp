#include "ops/test_matmul.hpp"
#include "minfer/ops/cpu_ops.hpp"

TestMatmul::TestMatmul(const std::string& name) : TestBase(name) {}

// TO-DO: improve test cases
void TestMatmul::test_identity() {
    float actual[5] = {0.0f};

    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float weight[25] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
    };
    int size = 5;

    float expected[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}; // i.e. equal to input

    cpu::matmul<float, float_tag>(actual, input, weight, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Identity matrix mult.");
}

void TestMatmul::test_square() {
    float actual[5] = {0.0f};

    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float weight[25] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -2.0f, -2.0f, -2.0f, -2.0f, -2.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };
    int size = 5;

    float expected[5] = {15.0f, 15.0f, -30.0f, 15.0f, 15.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Square matrix mult.");
}

void TestMatmul::test_rectangular() {
    float actual[3] = {0.0f};

    float input[4] = {1.0f, 2.0f, -3.0f, -4.0f};
    float weight[12] = {
        1.0f, -2.0f, 1.0f, -3.0f,
        3.0f, 4.0f, -5.0f, 6.0f,
        -2.0f, 2.0f, -2.0f, 2.0f
    };
    int d_out = 3;
    int d_in = 4;

    float expected[3] = {6.0f, 2.0f, 0.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, d_out, 1e-6f, "Rectangular matrix mult.");
}

void TestMatmul::test_zero_matrix() {
    float actual[3] = {0.0f};

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[12] = {0.0f}; // all zeros
    int d_out = 3;
    int d_in = 4;

    float expected[3] = {0.0f, 0.0f, 0.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, d_out, 1e-6f, "Zero matrix mult.");
}

void TestMatmul::test_zero_vector() {
    float actual[3] = {0.0f};

    float input[3] = {0.0f, 0.0f, 0.0f};
    float weight[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    int size = 3;

    float expected[3] = {0.0f, 0.0f, 0.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, size, size);

    assert_arrays_equal(expected, actual, size, 1e-6f, "Zero vector mult.");
}

void TestMatmul::test_single_element() {
    float actual[1] = {0.0f};

    float input[1] = {5.0f};
    float weight[1] = {3.0f};
    int d_out = 1;
    int d_in = 1;

    float expected[1] = {15.0f};

    cpu::matmul<float,float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, 1, 1e-6f, "Single element mult.");
}

void TestMatmul::test_large_values() {
    float actual[2] = {0.0f};

    float input[2] = {1000.0f, -2000.0f};
    float weight[4] = {
        0.001f, 0.002f,   // row 0
        -0.001f, 0.001f   // row 1
    };
    int d_out = 2;
    int d_in = 2;

    float expected[2] = {-3.0f, -3.0f};

    cpu::matmul<float, float_tag>(actual, input, weight, d_out, d_in);

    assert_arrays_equal(expected, actual, 2, 1e-3f, "Large value mult.");
}

// testing that the half_to_float and float_to_half functions work properly
void TestMatmul::test_fp16_conversion() {
    
    float test_values[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 3.14159f, -2.718f};
    size_t num_values = sizeof(test_values) / sizeof(test_values[0]);
    
    for (size_t i = 0; i < num_values; i++) {
        float original = test_values[i];
        fp16_t fp16_bits = float_to_half(original);
        float converted_back = half_to_float(fp16_bits);
        
        // FP16 limited precision => pick reasonable tolerance
        // ~3 decimal digits of precision
        float tolerance = std::max(1e-3f, std::abs(original) * 1e-3f);
        
        assert_equal(original, converted_back, tolerance, 
                    "FP16 round-trip conversion for value " + std::to_string(original));
    }
    
    fp16_t known_patterns[] = {
        0x0000, // +0.0
        0x8000, // -0.0  
        0x3C00, // +1.0
        0xBC00, // -1.0
        0x4000, // +2.0
        0x3800, // +0.5
    };
    float expected_values[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f};
    
    for (size_t i=0; i<6; i++) {
        float result = half_to_float(known_patterns[i]);
        assert_equal(expected_values[i], result, 1e-6f,
                    "Known FP16 bit pattern 0x" + std::to_string(known_patterns[i]));
    }
}

// test that FP16 matmul produces similar results to F32 matmul
void TestMatmul::test_fp16_matmul() {
    
    const int d_out = 3;
    const int d_in = 4;
    
    float input[4] = {1.0f, 2.0f, -0.5f, 1.5f}; // F32
    
    float weight_f32[12] = {
        1.0f, -0.5f, 2.0f, 0.25f,
        0.5f, 1.0f, -1.0f, 0.75f,
        -1.0f, 0.5f, 1.5f, -0.5f
    };
    
    fp16_t weight_fp16[12];
    for (size_t i=0; i<12; i++) {
        weight_fp16[i] = float_to_half(weight_f32[i]);
    }
    
    float result_f32[3] = {0.0f};
    float result_fp16[3] = {0.0f};
    
    cpu::matmul<float, float_tag>(result_f32, input, weight_f32, d_out, d_in);
    cpu::matmul<fp16_t, fp16_tag>(result_fp16, input, weight_fp16, d_out, d_in);
    
    for (int i=0; i<d_out; i++) {
        float tolerance = std::max(1e-2f, std::abs(result_f32[i]) * 1e-2f);
        assert_equal(result_f32[i], result_fp16[i], tolerance,
                    "FP16 vs F32 matmul result at index " + std::to_string(i));
    }
}

// testing that the half_to_float and float_to_half functions work properly
void TestMatmul::test_bf16_conversion() {
    
    float test_values[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 3.14159f, -2.718f};
    size_t num_values = sizeof(test_values) / sizeof(test_values[0]);
    
    for (size_t i = 0; i < num_values; i++) {
        float original = test_values[i];
        bf16_t bf16_bits = float_to_half(original);
        float converted_back = half_to_float(bf16_bits);
        
        // BF16 limited precision => pick reasonable tolerance
        // ~2 decimal digits of precision
        float tolerance = std::max(1e-2f, std::abs(original) * 1e-2f);
        
        assert_equal(original, converted_back, tolerance, 
                    "BF16 round-trip conversion for value " + std::to_string(original));
    }
    
    bf16_t known_patterns[] = {
        0x0000, // +0.0
        0x8000, // -0.0  
        0x3C00, // +1.0
        0xBC00, // -1.0
        0x4000, // +2.0
        0x3800, // +0.5
    };
    float expected_values[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f};
    
    for (size_t i=0; i<6; i++) {
        float result = half_to_float(known_patterns[i]);
        assert_equal(expected_values[i], result, 1e-6f,
                    "Known BF16 bit pattern 0x" + std::to_string(known_patterns[i]));
    }
}

// test that BF16 matmul produces similar results to F32 matmul
void TestMatmul::test_bf16_matmul() {
    
    const int d_out = 3;
    const int d_in = 4;
    
    float input[4] = {1.0f, 2.0f, -0.5f, 1.5f}; // F32
    
    float weight_f32[12] = {
        1.0f, -0.5f, 2.0f, 0.25f,
        0.5f, 1.0f, -1.0f, 0.75f,
        -1.0f, 0.5f, 1.5f, -0.5f
    };
    
    fp16_t weight_bf16[12];
    for (size_t i=0; i<12; i++) {
        weight_bf16[i] = float_to_half(weight_f32[i]);
    }
    
    float result_f32[3] = {0.0f};
    float result_bf16[3] = {0.0f};
    
    cpu::matmul<float, float_tag>(result_f32, input, weight_f32, d_out, d_in);
    cpu::matmul<bf16_t, bf16_tag>(result_bf16, input, weight_bf16, d_out, d_in);
    
    for (int i=0; i<d_out; i++) {
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
    test_single_element();
    test_large_values();
    test_fp16_conversion();
    test_fp16_matmul();
    test_bf16_conversion();
    test_bf16_matmul();
}