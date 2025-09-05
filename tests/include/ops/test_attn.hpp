#pragma once

#include "base/test_base.hpp"

class TestAttn : public TestBase {
private:
    void test_single_position();
    void test_multiple_positions();
    void test_attention_weights();
    void test_output_computation();

public:
    explicit TestAttn(const std::string& name);
    void run_all_tests() override;
};