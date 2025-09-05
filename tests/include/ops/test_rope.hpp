#pragma once

#include "base/test_base.hpp"

class TestRope : public TestBase {
private:
    void test_zero_position();
    void test_single_head();
    void test_multiple_heads();
    void test_partial_rotary();
    void test_input_to_input();

public:
    explicit TestRope(const std::string& name);
    void run_all_tests() override;
};