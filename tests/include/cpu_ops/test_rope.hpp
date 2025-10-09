#pragma once

#include "base/test_base.hpp"

class TestRope : public TestBase {
private:
    void test_il_rope_zero_position();
    void test_il_rope_single_head();
    void test_il_rope_partial_rotary();
    void test_neox_rope_zero_position();
    void test_neox_rope_single_head();
    void test_neox_rope_partial_rotary();
    void test_in_place_operation();
    void test_multiple_heads();

public:
    explicit TestRope(const std::string& name);
    void run_all_tests() override;
};