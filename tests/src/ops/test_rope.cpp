#include "ops/test_rope.hpp"
#include "minfer/ops/cpu_ops.hpp"

#include <cmath>

TestRope::TestRope(const std::string& name) : TestBase(name) {}

// TO-DO: test il_rope even though it isn't being used
void TestRope::test_zero_position() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 0;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // position=0 implies angle=0. cos(0)=1, sin(0)=0
    // output=input
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert_arrays_equal(expected, output, d_flat, 1e-6f, "RoPE at pos. 0");
}

// test single-head rotation at pos. 1
void TestRope::test_single_head() {
    float input[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float output[4] = {0.0f};
    int d_flat = 4;
    int d_head = 4;
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // pair (0,2): 
    // freq=10000^(-2*0/4)=1, angle=1*1=1
    // output[0]=cos(1)*1-sin(1)*0=cos(1)
    // output[2]=sin(1)*1+cos(1)*0=sin(1)
    
    // pair (1,3): 
    // freq=10000^(-2*1/4)=0.01, angle=0.01*1=0.01
    // output[1]=cos(0.01)*0-sin(0.01)*1=-sin(0.01)
    // output[3]=sin(0.01)*0+cos(0.01)*1=cos(0.01)

    float expected[4] = {std::cos(1.0f), -std::sin(0.01f), std::sin(1.0f) , std::cos(0.01f)};
    assert_arrays_equal(expected, output, d_flat, 1e-6f, "RoPE single head");
}

// test rope at pos. 1 for multiple heads
void TestRope::test_multiple_heads() {
    float input[8] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float output[8] = {0.0f};
    int d_flat = 8;
    int d_head = 4; // 2 heads, 4 dims each
    int d_rotary = 4;
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // Each head rotated same way since d_rotary=d_head
    // Head 0: idxs 0-3, Head 1: idxs 4-7
    
    assert_true(std::abs(output[0] - output[4]/2.0f) < 1e-6f, "Consistent rotation across heads (idx 0,4)");
    assert_true(std::abs(output[1] - output[5]/2.0f) < 1e-6f, "Consistent rotation across heads (idx 1,5)");
    assert_true(std::abs(output[2] - output[6]/2.0f) < 1e-6f, "Consistent rotation across heads (idx 2,6)");
    assert_true(std::abs(output[3] - output[7]/2.0f) < 1e-6f, "Consistent rotation across heads (idx 3,7)");
}

// testing partial rotations on multiple heads
void TestRope::test_partial_rotary() {
    float input[12] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[12] = {0.0f};
    int d_flat = 12; // 2 heads
    int d_head = 6;
    int d_rotary = 4; // just first 4 dims get rotated per head
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::neox_rope(output, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // first 4 dims of first head rotated
    assert_true(output[0] != input[0] || output[2] != input[2], "First pair of 1st head rotated");
    assert_true(output[1] != input[1] || output[3] != input[3], "Second pair of 1st head rotated");
    
    // last 2 dims of first head unchanged
    assert_equal(input[4], output[4], 1e-6f, "Dim. 4 (1st head) unchanged");
    assert_equal(input[5], output[5], 1e-6f, "Dim. 5 (1st head) unchanged");

    // first 4 dims of second head rotated
    assert_true(output[6] != input[6] || output[8] != input[8], "First pair of 2nd head rotated");
    assert_true(output[7] != input[7] || output[9] != input[9], "Second pair of 2nd head rotated");
    
    // last 2 dims of second head unchanged
    assert_equal(input[10], output[10], 1e-6f, "Dim. 10 (2nd head) unchanged");
    assert_equal(input[11], output[11], 1e-6f, "Dim. 11 (2nd head) unchanged");
    
    // the 2 heads rotated identically
    for (int i=0; i<6; ++i) {
        assert_equal(output[i], output[i+6], 1e-6f, "The two heads are rotated identically");
    }
}

void TestRope::test_input_to_input() {
    float input[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int d_flat = 6; // one head
    int d_head = 6;
    int d_rotary = 4; // just first 4 dims get rotated
    float freq_base = 10000.0f;
    int pos = 1;
    
    cpu::neox_rope(input, input, d_flat, d_head, d_rotary, freq_base, pos);
    
    // first 4 dims rotated
    assert_true(input[0] != 1.0f || input[2] != 1.0f, "First pair rotated");
    assert_true(input[1] != 1.0f || input[3] != 1.0f, "Second pair rotated");
    
    // last 2 dims unchanged
    assert_equal(input[4], 1.0f, 1e-6f, "Dim. 4 unchanged");
    assert_equal(input[5], 1.0f, 1e-6f, "Dim. 5 unchanged");
}

void TestRope::run_all_tests() {
    test_zero_position();
    test_single_head();
    test_multiple_heads();
    test_partial_rotary();
    test_input_to_input();
}