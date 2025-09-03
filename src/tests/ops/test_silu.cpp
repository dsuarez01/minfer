#include "tests/ops/test_silu.hpp"
#include "common/ops/cpu_ops.hpp"

#include <cmath>

TestSilu::TestSilu(const std::string& name) : TestBase(name) {}

 // silu(x) = x / (1 + exp(-x))
void TestSilu::test_zero() {
    float result = cpu::silu(0.0f);
    float expected = 0.0f;
    assert_equal(expected, result, 1e-6f, "SiLU at zero");
}

void TestSilu::test_positive() {
    float x = 1.0f;
    float result = cpu::silu(x);
    float expected = x / (1.0f + std::exp(-x));
    assert_equal(expected, result, 1e-6f, "SiLU eval. at x=1");
    
    x = 2.0f;
    result = cpu::silu(x);
    expected = x / (1.0f + std::exp(-x));
    assert_equal(expected, result, 1e-6f, "SiLU eval. at x=2");
}

void TestSilu::test_negative() {
    float x = -1.0f;
    float result = cpu::silu(x);
    float expected = x / (1.0f + std::exp(-x));
    assert_equal(expected, result, 1e-6f, "SiLU eval. at x=-1");
    
    x = -2.0f;
    result = cpu::silu(x);
    expected = x / (1.0f + std::exp(-x));
    assert_equal(expected, result, 1e-6f, "SiLU eval. at x=-2");
}

void TestSilu::run_all_tests() {
    test_zero();
    test_positive();
    test_negative();
}