#include "cpu_ops/test_rope.hpp"
#include "minfer/ops/kernels.hpp"

#include <cmath>

TestRope::TestRope(const std::string& name) : TestBase(name) {}

void TestRope::test_il_rope_zero_position() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f};
    int head_idx = 0;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    il_rope(output, input, head_idx, d_head, d_rotary, freq_base, pos);
    
    // at pos 0, all angles are 0: cos(0)=1, sin(0)=0
    // output = input here
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert_arrays_equal(expected, output, d_head, 1e-6f, "IL RoPE at pos 0");
}

void TestRope::test_il_rope_single_head() {
    float input[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float output[4] = {0.0f};
    int head_idx = 0;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    il_rope(output, input, head_idx, d_head, d_rotary, freq_base, pos);
    
    // Pair 0: freq = 1/10000^(2*0/4) = 1, angle = 1*1 = 1
    // output[0] = cos(1)*1 - sin(1)*0 = cos(1)
    // output[1] = sin(1)*1 + cos(1)*0 = sin(1)
    
    // Pair 1: freq = 1/10000^(2*1/4) = 1/100, angle = 1/100
    // output[2] = cos(0.01)*0 - sin(0.01)*1 = -sin(0.01)
    // output[3] = sin(0.01)*0 + cos(0.01)*1 = cos(0.01)
    
    float expected[4] = {std::cos(1.0f), std::sin(1.0f), -std::sin(0.01f), std::cos(0.01f)};
    assert_arrays_equal(expected, output, d_head, 1e-6f, "IL RoPE single head");
}

void TestRope::test_il_rope_partial_rotary() {
    float input[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int head_idx = 0;
    int d_head = 6;
    int d_rotary = 4; // first 4 dims get rotated (2 pairs)
    float freq_base = 10000.0f;
    int pos = 1;
    
    il_rope(output, input, head_idx, d_head, d_rotary, freq_base, pos);
    
    // first 4 dims rotated (2 pairs)
    assert_true(output[0] != 1.0f || output[1] != 1.0f, "First pair rotated");
    assert_true(output[2] != 1.0f || output[3] != 1.0f, "Second pair rotated");
    
    // last 2 dims unchanged
    assert_equal(input[4], output[4], 1e-6f, "Dim 4 unchanged");
    assert_equal(input[5], output[5], 1e-6f, "Dim 5 unchanged");
}

void TestRope::test_neox_rope_zero_position() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f};
    int head_idx = 0;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    neox_rope(output, input, head_idx, d_head, d_rotary, freq_base, pos);
    
    // at pos 0, all angles are 0
    // output = input
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert_arrays_equal(expected, output, d_head, 1e-6f, "NeoX RoPE at pos 0");
}

void TestRope::test_neox_rope_single_head() {
    float input[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float output[4] = {0.0f};
    int head_idx = 0;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    neox_rope(output, input, head_idx, d_head, d_rotary, freq_base, pos);
    
    // Pair (0,2): pair_idx=0, freq = 1/10000^(2*0/4) = 1, angle = 1
    // output[0] = cos(1)*1 - sin(1)*0 = cos(1)
    // output[2] = sin(1)*1 + cos(1)*0 = sin(1)
    
    // Pair (1,3): pair_idx=1, freq = 1/10000^(2*1/4) = 0.01, angle = 0.01
    // output[1] = cos(0.01)*0 - sin(0.01)*1 = -sin(0.01)
    // output[3] = sin(0.01)*0 + cos(0.01)*1 = cos(0.01)
    
    float expected[4] = {std::cos(1.0f), -std::sin(0.01f), std::sin(1.0f), std::cos(0.01f)};
    assert_arrays_equal(expected, output, d_head, 1e-6f, "NeoX RoPE single head");
}

void TestRope::test_neox_rope_partial_rotary() {
    float input[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int head_idx = 0;
    int d_head = 6;
    int d_rotary = 4; // d_rotary/2 = 2 pairs rotated
    float freq_base = 10000.0f;
    int pos = 1;
    
    neox_rope(output, input, head_idx, d_head, d_rotary, freq_base, pos);
    
    // first d_rotary/2 = 2 pairs are rotated: (0,2) and (1,3)
    assert_true(output[0] != 1.0f || output[2] != 1.0f, "Pair (0,2) rotated");
    assert_true(output[1] != 1.0f || output[3] != 1.0f, "Pair (1,3) rotated");
    
    // dims 4,5 unchanged
    assert_equal(input[4], output[4], 1e-6f, "Dim 4 unchanged");
    assert_equal(input[5], output[5], 1e-6f, "Dim 5 unchanged");
}

void TestRope::test_in_place_operation() {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int head_idx = 0;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    // test IL RoPE in-place at pos=0 (should be identity)
    il_rope(data, data, head_idx, d_head, d_rotary, freq_base, pos);
    assert_arrays_equal(expected, data, d_head, 1e-6f, "IL RoPE in-place at pos 0");
    
    // reset, test NeoX RoPE in-place at pos=0
    for (int i=0; i<4; i++) data[i] = expected[i];
    neox_rope(data, data, head_idx, d_head, d_rotary, freq_base, pos);
    assert_arrays_equal(expected, data, d_head, 1e-6f, "NeoX RoPE in-place at pos 0");
}

void TestRope::run_all_tests() {
    test_il_rope_zero_position();
    test_il_rope_single_head();
    test_il_rope_partial_rotary();
    test_neox_rope_zero_position();
    test_neox_rope_single_head();
    test_neox_rope_partial_rotary();
    test_in_place_operation();
}