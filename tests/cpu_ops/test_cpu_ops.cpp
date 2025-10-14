#include "base/test_base.hpp"
#include "cpu_ops/test_matmul.hpp"
#include "cpu_ops/test_silu.hpp"
#include "cpu_ops/test_softmax.hpp"
#include "cpu_ops/test_rmsnorm.hpp"
#include "cpu_ops/test_rope.hpp"
#include "cpu_ops/test_attn.hpp"
#include "cpu_ops/test_route.hpp"

#include <memory>
#include <vector>
#include <iostream>

int main() {
    std::vector<std::unique_ptr<TestBase>> all_tests;
    
    all_tests.push_back(std::make_unique<TestMatmul>("Matmul Tests"));
    all_tests.push_back(std::make_unique<TestSilu>("SiLU Tests"));
    all_tests.push_back(std::make_unique<TestSoftmax>("Softmax Tests"));
    all_tests.push_back(std::make_unique<TestRMSNorm>("RMSNorm Tests"));
    all_tests.push_back(std::make_unique<TestRope>("RoPE Tests"));
    all_tests.push_back(std::make_unique<TestAttn>("Attn. Tests"));
    all_tests.push_back(std::make_unique<TestRoute>("Router (MoE) Tests"));


    bool all_passed = true;
    int total_suites = 0;
    int passed_suites = 0;
    
    for (auto& test : all_tests) {
        total_suites++;
        test->run_all_tests();
        test->print_summary();
        if (test->all_passed()) {
            passed_suites++;
        } else {
            all_passed = false;
        }
        std::cout << std::endl;
    }
    
    std::cout << "=== OVERALL SUMMARY ===" << std::endl;
    std::cout << "Test suites passed: " << passed_suites << "/" << total_suites << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}