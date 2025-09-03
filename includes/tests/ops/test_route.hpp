#pragma once

#include "tests/base/test_base.hpp"

class TestRoute : public TestBase {
private:
    void test_single_expert();
    void test_top_k_selection();
    void test_score_normalization();
    void test_expert_ordering();

public:
    explicit TestRoute(const std::string& name);
    void run_all_tests() override;
};