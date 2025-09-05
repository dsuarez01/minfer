#include "ops/test_rmsnorm.hpp"
#include "minfer/ops/cpu_ops.hpp"

#include <cmath>

TestRMSNorm::TestRMSNorm(const std::string& name) : TestBase(name) {}

void TestRMSNorm::test_unit_vector() {
    float input[4] = {3.0f, 4.0f, 0.0f, 0.0f};
    float weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[4] = {0.0f};
    float eps = 1e-6f;
    
    cpu::rmsnorm(output, input, weights, 4, eps);
    
    float rms = std::sqrtf(25.0f/4.0f + eps);
    float expected[4] = {3.0f/rms * 1.0f, 4.0f/rms * 1.0f, 0.0f, 0.0f};
    
    assert_arrays_equal(expected, output, 4, 1e-6f, "Unit vector rmsnorm");
}

void TestRMSNorm::test_scaling() {
    float input[3] = {1.0f, 2.0f, 3.0f};
    float weights[3] = {2.0f, 0.5f, 1.0f}; // different scale factors
    float output[3] = {0.0f};
    float eps = 1e-6f;
    
    cpu::rmsnorm(output,input, weights, 3, eps);
    
    float rms = std::sqrt(14.0f/3.0f + eps);
    float expected[3] = {1.0f/rms * 2.0f, 2.0f/rms * 0.5f, 3.0f/rms * 1.0f};
    
    assert_arrays_equal(expected, output, 3, 1e-6f, "Scaled weights rmsnorm");
}

void TestRMSNorm::test_zero_input() {
    float input[3] = {0.0f, 0.0f, 0.0f};
    float weights[3] = {1.0f, 1.0f, 1.0f};
    float output[3] = {0.0f};
    float eps = 1e-6f;
    
    cpu::rmsnorm(output, input, weights, 3, eps);
    
    float expected[3] = {0.0f, 0.0f, 0.0f};
    
    assert_arrays_equal(expected, output, 3, 1e-6f, "Zero input rmsnorm");
}

void TestRMSNorm::test_uniform_weights() {
    float input[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[4] = {0.0f};
    float eps = 1e-6f;
    
    cpu::rmsnorm(output, input, weights, 4, eps);
    
    float rms = std::sqrtf((4.0f * 4)/4 + eps);
    float expected[4] = {2.0f/rms * 1.0f, 2.0f/rms * 1.0f, 2.0f/rms * 1.0f, 2.0f/rms * 1.0f};
    
    assert_arrays_equal(expected, output, 4, 1e-6f, "Uniform input rmsnorm");
}

void TestRMSNorm::run_all_tests() {
    test_unit_vector();
    test_scaling();
    test_zero_input();
    test_uniform_weights();
}