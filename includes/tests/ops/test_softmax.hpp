#pragma once

#include "tests/base/test_base.hpp"

class TestSoftmax : public TestBase {
private:
    void test_uniform();
    void test_one_hot();
    void test_large_values();
    void test_sum_to_one();

public:
    explicit TestSoftmax(const std::string& name);
    void run_all_tests() override;
};