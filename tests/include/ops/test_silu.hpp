#pragma once

#include "base/test_base.hpp"

class TestSilu : public TestBase {
private:
    void test_zero();
    void test_positive();
    void test_negative();

public:
    explicit TestSilu(const std::string& name);
    void run_all_tests() override;
};