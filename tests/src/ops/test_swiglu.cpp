#include "ops/test_swiglu.hpp"
#include "minfer/ops/cpu_ops.hpp"

#include <cmath>

TestSwiglu::TestSwiglu(const std::string& name) : TestBase(name) {}

void TestSwiglu::test_basic_computation() {
    int d_model = 2;
    int d_ff = 3;
    
    float x_in[2] = {1.0f, 2.0f};
    float w_gate[6] = {
        1.0f, 0.0f, // gate neuron 0
        0.0f, 1.0f,  // gate neuron 1  
        1.0f, 1.0f  // gate neuron 2
    };
    float w_up[6] = {
        2.0f, 0.0f,  // up neuron 0
        0.0f, 2.0f,   // up neuron 1
        1.0f, 1.0f   // up neuron 2
    };
    float w_down[6] = {
        1.0f, 0.0f, 0.0f,   // output 0
        0.0f, 2.0f, 0.0f   // output 1
    };
    
    float exp_buf[2] = {0.0f};
    float gate_buf[3] = {0.0f};
    float up_buf[3] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    // basic matmul w/ w_up for up buf, silu(gate(x))*up(x) for gate buf
    // output is down(up(x) * silu(gate(x)))
    float expected_out[2] = {1.0f*(cpu::silu(1.0f)*2.0f),2.0f*(cpu::silu(2.0f)*4.0f)};
    float expected_gate[3] = {cpu::silu(1.0f)*2.0f, cpu::silu(2.0f)*4.0f, cpu::silu(3.0f)*3.0f};
    float expected_up[3] = {2.0f, 4.0f, 3.0f};
    
    assert_arrays_equal(expected_out, exp_buf, d_model, 1e-6f, "Final output: basic");
    assert_arrays_equal(expected_gate, gate_buf, d_ff, 1e-6f, "Gate buf: basic");
    assert_arrays_equal(expected_up, up_buf, d_ff, 1e-6f, "Up buf: basic");
}

void TestSwiglu::test_zero_input() {
    int d_model = 2;
    int d_ff = 2;
    
    float x_in[2] = {0.0f, 0.0f};
    
    float w_gate[4] = {
        1.0f, 1.0f, 
        1.0f, 1.0f
    };
    float w_up[4] = {
        1.0f, 1.0f, 
        1.0f, 1.0f
    };
    float w_down[4] = {
        1.0f, 0.0f, 
        0.0f, 1.0f
    };
    
    float exp_buf[2] = {0.0f};
    float gate_buf[2] = {0.0f};
    float up_buf[2] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    // zero input => zero output
    float expected_out[2] = {0.0f, 0.0f};
    float expected_gate[2] = {0.0f, 0.0f}; // silu(0.0f) * 0.0f
    float expected_up[2] = {0.0f, 0.0f};
    assert_arrays_equal(expected_out, exp_buf, d_model, 1e-6f, "Final output: zero input => zero output");
    assert_arrays_equal(expected_gate, gate_buf, d_ff, 1e-6f, "Gate buf: zero input => zero output");
    assert_arrays_equal(expected_up, up_buf, d_ff, 1e-6f, "Up buf: zero input => zero output");
}

void TestSwiglu::test_identity_weights() {
    int d_model = 2;
    int d_ff = 2;
    
    float x_in[2] = {1.0f, 2.0f};
    
    // identity
    float w_gate[4] = {
        1.0f, 0.0f, 
        0.0f, 1.0f
    };
    float w_up[4] = {
        1.0f, 0.0f, 
        0.0f, 1.0f
    };
    float w_down[4] = {
        1.0f, 0.0f, 
        0.0f, 1.0f
    };
    
    float exp_buf[2] = {0.0f};
    float gate_buf[2] = {0.0f};
    float up_buf[2] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    // Up=input=[1, 2]
    // After SiLU and multiplication: [silu(1)*1, silu(2)*2] 
    float expected_out[2] = {1.0f* (cpu::silu(1.0f) * 1.0f), 1.0f*(cpu::silu(2.0f) * 2.0f)};
    float expected_gate[2] = {cpu::silu(1.0f) * 1.0f, cpu::silu(2.0f) * 2.0f};
    float expected_up[2] = {1.0f, 2.0f};

    assert_arrays_equal(expected_out, exp_buf, d_model, 1e-6f, "Final output: identity weights");
    assert_arrays_equal(expected_gate, gate_buf, d_ff, 1e-6f, "Gate buf: identity weights");
    assert_arrays_equal(expected_up, up_buf, d_ff, 1e-6f, "Up buf: identity weights");
}

void TestSwiglu::test_scalar() {
    int d_model = 1;
    int d_ff = 1;
    
    float x_in[1] = {1.0f};
    float w_gate[1] = {2.0f};  // gate=2*1=2
    float w_up[1] = {3.0f};    // up=3*1=3
    float w_down[1] = {4.0f};
    
    float exp_buf[1] = {0.0f};
    float gate_buf[1] = {0.0f};
    float up_buf[1] = {0.0f};
    
    cpu::swiglu<float, float_tag>(x_in, exp_buf, gate_buf, up_buf, w_gate, w_up, w_down, d_ff, d_model);
    
    float expected_out = 4.0f * (cpu::silu(2.0f) * 3.0f);
    float expected_gate = cpu::silu(2.0f) * 3.0f;
    float expected_up = 3.0f;
    assert_equal(expected_out, exp_buf[0], 1e-6f, "Final output: scalar");
    assert_equal(expected_gate, gate_buf[0], 1e-6f, "Gate buf: scalar");
    assert_equal(expected_up, up_buf[0], 1e-6f, "Up buf: scalar");
}

void TestSwiglu::run_all_tests() {
    test_basic_computation();
    test_zero_input();
    test_identity_weights();
    test_scalar();
}