#include "ops/test_swiglu.hpp"
#include "minfer/ops/cpu_ops.hpp"

#include <cmath>

TestSwiglu::TestSwiglu(const std::string& name) : TestBase(name) {}

void TestSwiglu::test_basic_computation() {
    int d_model = 16;
    int d_ff = 32;
    
    float x_in[16];
    for (int i = 0; i < 16; i++) {
        x_in[i] = (i % 2 == 0) ? 1.0f : 2.0f;
    }
    
    float w_gate[512];  // 32 × 16
    float w_up[512];    // 32 × 16  
    float w_down[512];  // 16 × 32
    
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            w_gate[i * 16 + j] = (i < 16 && i == j) ? 1.0f : 0.0f; // simple diagonal pattern for gate
            // 2x scaling for up
            w_up[i * 16 + j] = (i < 16 && i == j) ? 2.0f : 0.0f;
        }
    }
    
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            w_down[i * 32 + j] = (i == j % 16) ? 1.0f : 0.0f; // identity-like for down projection
        }
    }
    
    float exp_buf[16] = {0.0f};
    float gate_buf[32] = {0.0f};
    float up_buf[32] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    // expected values for first 16 elements
    float expected_gate[32] = {0.0f};
    float expected_up[32] = {0.0f};
    for (int i = 0; i < 16; i++) {
        expected_up[i] = x_in[i] * 2.0f; // 2x scaling
        expected_gate[i] = cpu::silu(x_in[i]) * expected_up[i];
    }
    
    assert_arrays_equal(expected_up, up_buf, 16, 1e-6f, "Up buf: basic computation");
    assert_arrays_equal(expected_gate, gate_buf, 16, 1e-6f, "Gate buf: basic computation");
}

void TestSwiglu::test_zero_input() {
    int d_model = 16;
    int d_ff = 16;
    
    float x_in[16] = {0.0f};
    
    float w_gate[256], w_up[256], w_down[256];
    for (int i = 0; i < 256; i++) {
        w_gate[i] = 1.0f;
        w_up[i] = 1.0f;  
        w_down[i] = 1.0f;
    }
    
    float exp_buf[16] = {0.0f};
    float gate_buf[16] = {0.0f};
    float up_buf[16] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    float expected_out[16] = {0.0f};
    float expected_gate[16] = {0.0f};
    float expected_up[16] = {0.0f};
    
    assert_arrays_equal(expected_out, exp_buf, d_model, 1e-6f, "Zero input => zero output");
    assert_arrays_equal(expected_gate, gate_buf, d_ff, 1e-6f, "Zero gate buffer");
    assert_arrays_equal(expected_up, up_buf, d_ff, 1e-6f, "Zero up buffer");
}

void TestSwiglu::test_identity_weights() {
    int d_model = 16;
    int d_ff = 16;
    
    float x_in[16];
    for (int i = 0; i < 16; i++) {
        x_in[i] = i + 1.0f; // 1, 2, 3, ..., 16
    }
    
    // identity matrices
    float w_gate[256] = {0.0f};
    float w_up[256] = {0.0f};
    float w_down[256] = {0.0f};
    
    for (int i = 0; i < 16; i++) {
        w_gate[i * 16 + i] = 1.0f;
        w_up[i * 16 + i] = 1.0f;
        w_down[i * 16 + i] = 1.0f;
    }
    
    float exp_buf[16] = {0.0f};
    float gate_buf[16] = {0.0f};
    float up_buf[16] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    float expected_up[16];
    float expected_gate[16];
    float expected_out[16];
    
    for (int i = 0; i < 16; i++) {
        expected_up[i] = x_in[i];  // identity transformation
        expected_gate[i] = cpu::silu(x_in[i]) * x_in[i];
        expected_out[i] = expected_gate[i];  // identity down proj.
    }
    
    assert_arrays_equal(expected_out, exp_buf, d_model, 1e-6f, "Final output: identity weights");
    assert_arrays_equal(expected_gate, gate_buf, d_ff, 1e-6f, "Gate buf: identity weights");
    assert_arrays_equal(expected_up, up_buf, d_ff, 1e-6f, "Up buf: identity weights");
}

void TestSwiglu::test_scalar() {
    // test with d_model=16, d_ff=16 but use only first element
    int d_model = 16;
    int d_ff = 16;
    
    float x_in[16] = {1.0f, 0.0f}; // only first element non-zero
    
    float w_gate[256] = {0.0f};
    float w_up[256] = {0.0f};
    float w_down[256] = {0.0f};
    
    // only first row/column active
    w_gate[0] = 2.0f;  // gate[0] = 2*1 = 2
    w_up[0] = 3.0f;    // up[0] = 3*1 = 3
    w_down[0] = 4.0f;  // down[0] = 4*gate[0] = 4*silu(2)*3
    
    float exp_buf[16] = {0.0f};
    float gate_buf[16] = {0.0f};
    float up_buf[16] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    float expected_out = 4.0f * (cpu::silu(2.0f) * 3.0f);
    float expected_gate = cpu::silu(2.0f) * 3.0f;
    float expected_up = 3.0f;
    
    assert_equal(expected_out, exp_buf[0], 1e-6f, "Final output: scalar-like");
    assert_equal(expected_gate, gate_buf[0], 1e-6f, "Gate buf: scalar-like");
    assert_equal(expected_up, up_buf[0], 1e-6f, "Up buf: scalar-like");
    
    for (int i = 1; i < 16; i++) { // other elements should be zero
        assert_equal(0.0f, exp_buf[i], 1e-6f, "Other outputs zero");
        assert_equal(0.0f, gate_buf[i], 1e-6f, "Other gate elements zero");
        assert_equal(0.0f, up_buf[i], 1e-6f, "Other up elements zero");
    }
}

void TestSwiglu::run_all_tests() {
    test_basic_computation();
    test_zero_input();
    test_identity_weights();
    test_scalar();
}