#include "tests/ops/test_softmax.hpp"
#include "common/ops/cpu_ops.hpp"

#include <cmath>

TestSoftmax::TestSoftmax(const std::string& name) : TestBase(name) {}

void TestSoftmax::test_uniform() {
    float input[3] = {0.0f, 0.0f, 0.0f};
    float output[3] = {0.0f};
    float expected[3] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f}; // uniform dist. weights
    
    cpu::softmax(output, input, 3);
    
    assert_arrays_equal(expected, output, 3, 1e-6f, "Uniform input softmax");
}

void TestSoftmax::test_one_hot() {
    float input[4] = {-1000000.0f, 0.0f, -1000000.0f, -1000000.0f};
    float output[4] = {0.0f};
    float expected[4] = {0.0f, 1.0f, 0.0f, 0.0f}; // approx. one-hot at index 1
    
    cpu::softmax(output, input, 4);
    
    assert_arrays_equal(expected, output, 4, 1e-6f, "One-hot softmax");
}

// numerical stability with large values
void TestSoftmax::test_large_values() {
    float input[3] = {1000.0f, 1001.0f, 999.0f};
    float output[3] = {0.0f};
    
    cpu::softmax(output, input, 3);
    
    // no nan, inf
    for (int i = 0; i < 3; i++) {
        assert_true(!std::isnan(output[i]), "No NaN in large val. softmax");
        assert_true(!std::isinf(output[i]), "No Inf in large val. softmax");
    }
    
    // max val gets most prob
    assert_true(output[1] > output[0] && output[1] > output[2], "Max input (at idx. 1) gets max probability");
}

void TestSoftmax::test_sum_to_one() {
    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float output[5] = {0.0f};
    
    cpu::softmax(output, input, 5);
    
    float sum = 0.0f;
    for (int i = 0; i < 5; i++) {
        sum += output[i];
    }
    
    assert_equal(1.0f, sum, 1e-6f, "Softmax probs. sum to 1");
}

void TestSoftmax::run_all_tests() {
    test_uniform();
    test_one_hot();
    test_large_values();
    test_sum_to_one();
}