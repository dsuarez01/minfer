#include "base/test_base.hpp"

#include <iostream>
#include <cmath>
#include <vector>

TestBase::TestBase(const std::string& name) : test_name(name) {}

void TestBase::assert_true(
    bool condition, 
    const std::string& message
) {
    total_tests++;
    if (condition) {
        passed_tests++;
        std::cout << "." << std::flush;
    } else {
        std::cout << "F" << std::flush;
        failures.push_back(message);
    }
}

void TestBase::assert_equal(
    float expected, 
    float actual, 
    float tolerance, 
    const std::string& message
) {
    bool passed = std::abs(expected-actual)<=tolerance;
    total_tests++;
    if (passed) {
        passed_tests++;
        std::cout << "." << std::flush;
    } else {
        std::cout << "F" << std::flush;
        std::string failure_msg = 
        message + " (expected: " + std::to_string(expected) 
                + ", got: " + std::to_string(actual)
                + ", diff: " + std::to_string(std::abs(expected-actual)) 
                + ")";
        failures.push_back(failure_msg);
    }
}

void TestBase::assert_arrays_equal(
    const float* expected,
    const float* actual,
    size_t size,
    float tolerance,
    const std::string& message
) {
    total_tests++;
    for (size_t i=0; i<size; i++) {
        if (std::abs(expected[i]-actual[i]) > tolerance) {
            std::cout << "F" << std::flush;
            std::string failure_msg = 
            message + " at index " + std::to_string(i) 
                    + " (expected: " + std::to_string(expected[i])
                    + ", got: " + std::to_string(actual[i]) 
                    + ")";
            failures.push_back(failure_msg);
            return;
        }
    }
    passed_tests++;
    std::cout << "." << std::flush;
}

void TestBase::print_summary() {
    std::cout << "\n\n=== " << test_name << " Summary ===" << std::endl;
    std::cout << "Passed: " << passed_tests << "/" << total_tests << std::endl;
    
    if (passed_tests == total_tests) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << (total_tests-passed_tests) << " tests failed!" << std::endl;
        
        if (!failures.empty()) {
            std::cout << "\nFAILURES:" << std::endl;
            for (size_t i=0; i < failures.size(); i++) {
                std::cout << (i+1) << ". " << failures[i] << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

bool TestBase::all_passed() const {
    return passed_tests == total_tests;
}