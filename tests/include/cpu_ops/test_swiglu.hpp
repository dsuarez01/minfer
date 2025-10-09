#pragma once

#include "base/test_base.hpp"

class TestSwiglu : public TestBase {
private:
    void test_basic_computation();
    void test_zero_input();
    void test_identity_weights();
    void test_scalar();

public:
    explicit TestSwiglu(const std::string& name);
    void run_all_tests() override;
};