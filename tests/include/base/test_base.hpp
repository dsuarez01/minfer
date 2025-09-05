#pragma once

#include <string>
#include <vector>

class TestBase {
private:
    std::string test_name;
    int passed_tests = 0;
    int total_tests = 0;
    std::vector<std::string> failures;
    
public:
    explicit TestBase(const std::string& name);
    virtual ~TestBase() = default;
    
    virtual void run_all_tests() = 0;
    
    void assert_true(bool condition, const std::string& message = "");
    void assert_equal(float expected, float actual, float tolerance = 1e-6f, const std::string& message = "");
    void assert_arrays_equal(const float* expected, const float* actual, size_t size, 
                           float tolerance = 1e-6f, const std::string& message = "");
    
    void print_summary();
    bool all_passed() const;
};