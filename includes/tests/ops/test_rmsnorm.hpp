#pragma once

#include "tests/base/test_base.hpp"

class TestRMSNorm : public TestBase {
private:
    void test_unit_vector();
    void test_scaling();
    void test_zero_input();
    void test_uniform_weights();

public:
    explicit TestRMSNorm(const std::string& name);
    void run_all_tests() override;
};