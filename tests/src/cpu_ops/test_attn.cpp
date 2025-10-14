#include "cpu_ops/test_attn.hpp"
#include "minfer/ops/kernels.hpp"

#include <cmath>

// TO-DO: improve test cases
TestAttn::TestAttn(const std::string& name) : TestBase(name) {}

void TestAttn::test_single_position() {
    int seq_len = 1;
    int d_head = 4;
    int kv_dim = 4;
    
    float q_head[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float kh[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float vh[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float att_scores[1] = {0.0f};
    float att_out[4] = {0.0f};
    
    attn(att_scores, att_out, q_head, kh, vh, seq_len, d_head, kv_dim);
    
    assert_equal(1.0f, att_scores[0], 1e-6f, "Single position attn weight");
    
    float expected_out[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    assert_arrays_equal(expected_out, att_out, 1*d_head, 1e-6f, "Single position attn output");
}

void TestAttn::test_multiple_positions() {
    int seq_len = 3;
    int d_head = 2;
    int kv_dim = 2;
    
    float q_head[2] = {1.0f, 0.0f};
    float kh[6] = {
        1.0f, 0.0f,  // pos 0: identical to query
        0.0f, 1.0f,  // pos 1: orthogonal to query
        2.0f, 0.0f   // pos 2: 2x query direction
    };
    float vh[6] = {
        1.0f, 1.0f,  // pos 0
        2.0f, 2.0f,  // pos 1
        3.0f, 3.0f   // pos 2
    };
    float att_scores[3] = {0.0f};
    float att_out[2] = {0.0f};
    
    attn(att_scores, att_out, q_head, kh, vh, seq_len, d_head, kv_dim);
    
    // attn scores sum to 1
    float sum = att_scores[0] + att_scores[1] + att_scores[2];
    assert_equal(1.0f, sum, 1e-6f, "Attention weights sum to 1");
    
    // pos. 2 has highest attn
    assert_true(att_scores[2] > att_scores[0], "Greater score for larger dot product");
    assert_true(att_scores[2] > att_scores[1], "Greater score for larger dot product");
    
    // pos. 1 has lowest attn
    assert_true(att_scores[1] < att_scores[0], "Lower score for orthogonal vectors");
}

void TestAttn::test_attention_weights() {
    int seq_len = 2;
    int d_head = 3;
    int kv_dim = 3;
    
    float q_head[3] = {1.0f, 1.0f, 1.0f};
    float kh[6] = {
        1.0f, 1.0f, 1.0f,  // pos 0: dot product = 3
        -1.0f, -1.0f, -1.0f // pos 1: dot product = -3
    };
    float vh[6] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    float att_scores[2] = {0.0f};
    float att_out[3] = {0.0f};
    
    attn(att_scores, att_out, q_head, kh, vh, seq_len, d_head, kv_dim);
    
    float abs_score = 1/sqrtf(d_head)*3;
    float scores[2] = {
        abs_score,
        -abs_score
    };
    float weights[2] = {
        std::expf(abs_score)/(std::expf(abs_score) + std::expf(-abs_score)),
        std::expf(-abs_score)/(std::expf(abs_score) + std::expf(-abs_score))
    };

    float expected[3] = {weights[0], weights[1], 0.0f};

    assert_arrays_equal(weights, att_scores, seq_len, 1e-6, "Weights check");
    assert_arrays_equal(expected, att_out, 1*d_head, 1e-6, "Attention output check");
}

void TestAttn::test_output_computation() {
    int seq_len = 2;
    int d_head = 2;
    int kv_dim = 2;
    
    float q_head[2] = {1.0f, 0.0f};
    float kh[4] = {
        1000.0f, 0.0f,  // very high score
        0.0f, 0.0f     // very low score
    };
    float vh[4] = {
        5.0f, 7.0f,   // pos 0 value
        100.0f, 200.0f // pos 1 value
    };
    float att_scores[2] = {0.0f};
    float att_out[2] = {0.0f};
    
    attn(att_scores, att_out, q_head, kh, vh, seq_len, d_head, kv_dim);
    
    // output (very) close to first value vector
    assert_true(std::abs(att_out[0] - 5.0f) < 0.1f, "Output dominated by high-attention position");
    assert_true(std::abs(att_out[1] - 7.0f) < 0.1f, "Output dominated by high-attention position");
}

void TestAttn::run_all_tests() {
    test_single_position();
    test_multiple_positions();
    test_attention_weights();
    test_output_computation();
}