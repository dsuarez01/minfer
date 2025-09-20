#include "ops/test_rope.hpp"
#include "minfer/ops/cpu_ops.hpp"

#include <cmath>

TestRope::TestRope(const std::string& name) : TestBase(name) {}

void TestRope::test_il_rope_zero_position() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    cpu::il_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // at pos 0, all angles are 0: cos(0)=1, sin(0)=0
    // output = input here
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert_arrays_equal(expected, output, d_flat, 1e-6f, "IL RoPE at pos 0");
}

void TestRope::test_il_rope_single_head() {
    float input[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float output[4] = {0.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::il_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // Pair (0,1): freq = 1/10000^(0/4) = 1, angle = 1*1 = 1
    // output[0] = cos(1)*1 - sin(1)*0 = cos(1)
    // output[1] = sin(1)*1 + cos(1)*0 = sin(1)
    
    // Pair (2,3): freq = 1/10000^(2/4) = 1/100, angle = 1/100
    // output[2] = cos(0.01)*0 - sin(0.01)*1 = -sin(0.01)
    // output[3] = sin(0.01)*0 + cos(0.01)*1 = cos(0.01)
    
    float expected[4] = {std::cos(1.0f), std::sin(1.0f), -std::sin(0.01f), std::cos(0.01f)};
    assert_arrays_equal(expected, output, d_flat, 1e-6f, "IL RoPE single head");
}

void TestRope::test_il_rope_partial_rotary() {
    float input[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[6] = {0.0f};
    int d_flat = 6;
    int d_head = 6;
    int d_rotary = 4; // first 4 dims get rotated
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::il_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // first 4 dims rotated
    assert_true(output[0] != 1.0f || output[1] != 1.0f, "First pair rotated");
    assert_true(output[2] != 1.0f || output[3] != 1.0f, "Second pair rotated");
    
    // last 2 dims unchanged
    assert_equal(input[4], output[4], 1e-6f, "Dim 4 unchanged");
    assert_equal(input[5], output[5], 1e-6f, "Dim 5 unchanged");
}

void TestRope::test_neox_rope_zero_position() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // at pos 0, all angles are 0
    // output = input
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert_arrays_equal(expected, output, d_flat, 1e-6f, "NeoX RoPE at pos 0");
}

void TestRope::test_neox_rope_single_head() {
    float input[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float output[4] = {0.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // Pair (0,2): i_0=0, freq = 1/10000^(2*0/4) = 1, angle = 1
    // output[0] = cos(1)*1 - sin(1)*0 = cos(1)
    // output[2] = sin(1)*1 + cos(1)*0 = sin(1)
    
    // Pair (1,3): i_0=1, freq = 1/10000^(2*1/4) = 0.01, angle = 0.01
    // output[1] = cos(0.01)*0 - sin(0.01)*1 = -sin(0.01)
    // output[3] = sin(0.01)*0 + cos(0.01)*1 = cos(0.01)
    
    float expected[4] = {std::cos(1.0f), -std::sin(0.01f), std::sin(1.0f), std::cos(0.01f)};
    assert_arrays_equal(expected, output, d_flat, 1e-6f, "NeoX RoPE single head");
}

void TestRope::test_neox_rope_partial_rotary() {
    float input[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[6] = {0.0f};
    int d_flat = 6;
    int d_head = 6;
    int d_rotary = 4; // only first d_rotary/2 = 2 pairs rotated
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // first d_rotary/2 = 4/2 = 2 pairs are rotated
    assert_true(output[0] != 1.0f || output[2] != 1.0f, "Pair (0,2) rotated");
    assert_true(output[1] != 1.0f || output[3] != 1.0f, "Pair (1,3) rotated");
    
    // dims 4,5 unchanged
    assert_equal(input[4], output[4], 1e-6f, "Dim 4 unchanged");
    assert_equal(input[5], output[5], 1e-6f, "Dim 5 unchanged");
}

void TestRope::test_in_place_operation() {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    // test IL RoPE in-place at pos=0 (should be identity)
    cpu::il_rope(data, data, d_flat, d_head, d_rotary, freq_base, pos);
    assert_arrays_equal(expected, data, d_flat, 1e-6f, "IL RoPE in-place at pos 0");
    
    // reset, test NeoX RoPE in-place at pos=0
    for (int i=0; i<4; i++) data[i] = expected[i];
    cpu::neox_rope(data, data, d_flat, d_head, d_rotary, freq_base, pos);
    assert_arrays_equal(expected, data, d_flat, 1e-6f, "NeoX RoPE in-place at pos 0");
}

void TestRope::test_multiple_heads() {
    float input[8] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float il_output[8] = {0.0f};
    float neox_output[8] = {0.0f};
    int d_flat = 8;
    int d_head = 4; // 2 heads, 4 dims each
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::il_rope(il_output, input, d_flat, d_head, d_rotary, freq_base, pos);
    cpu::neox_rope(neox_output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // each head is rotated in the exact same way
    // Head 0: indices 0-3, Head 1: indices 4-7
    // Head 1 input is 2x Head 0 input, etc. for the rest
     
    for (int i=0; i<4; i++) {
        assert_equal(il_output[i] * 2.0f, il_output[i+4], 1e-5f, 
                    "IL RoPE consistent across heads");
        assert_equal(neox_output[i] * 2.0f, neox_output[i+4], 1e-5f, 
                    "NeoX RoPE consistent across heads");
    }
}

void TestRope::run_all_tests() {
    test_il_rope_zero_position();
    test_il_rope_single_head();
    test_il_rope_partial_rotary();
    test_neox_rope_zero_position();
    test_neox_rope_single_head();
    test_neox_rope_partial_rotary();
    test_in_place_operation();
    test_multiple_heads();
}