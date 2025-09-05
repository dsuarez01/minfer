#pragma once

#include "base/test_base.hpp"

class TestMatmul : public TestBase {
private:
    void test_identity(); // square identity matrix 
    void test_square();
    void test_rectangular();
    void test_zero_matrix();
    void test_zero_vector();
    void test_single_element();
    void test_large_values();
    void test_fp16_conversion();
    void test_fp16_matmul();
    void test_bf16_conversion();
    void test_bf16_matmul();

public:
    explicit TestMatmul(const std::string& name);
    void run_all_tests() override;
};