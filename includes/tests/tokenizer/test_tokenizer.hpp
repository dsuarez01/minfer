#pragma once

#include "tests/base/test_base.hpp"
#include "common/models/qwen3/tokenizer.hpp"

#include <string>
#include <memory>

class TestTokenizer : public TestBase {
private:
    std::unique_ptr<Qwen3Tokenizer> tokenizer;
    
    void test_basic_roundtrip();
    void test_empty_string();
    void test_special_tokens();
    void test_unicode_characters();
    void test_mixed_content();
    void test_numbers_and_punctuation();
    void test_paste();
    
public:
    explicit TestTokenizer(const std::string& name);
    bool init_from_gguf(const std::string& gguf_path);
    void run_all_tests() override;
    void assert_roundtrip(const std::string& input, const std::string& test_name = "");
};