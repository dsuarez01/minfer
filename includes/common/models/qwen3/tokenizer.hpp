#pragma once

#include "common/base/tokenizer.hpp"
#include "minja/chat-template.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#define PCRE2_CODE_UNIT_WIDTH 8
#include "pcre2.h"


class Qwen3Tokenizer : public BaseTokenizer {
public:
    Qwen3Tokenizer(
        const std::vector<std::string>& tokens,
        const std::vector<std::string>& merges,
        const std::vector<uint32_t>& token_types,
        const std::string& chat_template,
        uint32_t eos_id,
        uint32_t pad_id
    );
    
    std::vector<uint32_t> encode(const std::string& text) override;
    std::string decode(const std::vector<uint32_t>& tokens) override;
    std::string decode_token(uint32_t token_id) override;
    
    std::string apply_chat_template(
        const std::vector<Message>& messages,
        const std::vector<Tool>& tools = {},
        bool add_generation_prompt = true
    ) override;

private:
    static constexpr uint32_t CONTROL_TOKEN = 3;
    static constexpr uint32_t USER_DEFINED_TOKEN = 4;
    static constexpr uint32_t UNUSED_TOKEN = 5;
    // TO-DO: figure out how to handle token type 6

    struct MergeRule {
        std::string left;
        std::string right;
        std::string merged;
    };
    
    std::vector<std::string> _vocab;
    std::vector<uint32_t> _token_types;
    std::unordered_set<std::string> _special_tokens;
    std::unique_ptr<minja::chat_template> _chat_template;
    std::vector<MergeRule> _merge_rules;
    std::unordered_map<std::string, uint32_t> _vocab_to_id;
    
    // deleters to manage lifetime of PCRE2 objects
    struct PCRE2Deleter {
        void operator()(pcre2_code* ptr) const { if (ptr) pcre2_code_free(ptr); }
    };
    std::unique_ptr<pcre2_code, PCRE2Deleter> _compiled_pattern;

    struct PCRE2MatchDataDeleter {
        void operator()(pcre2_match_data* ptr) const { 
            if (ptr) pcre2_match_data_free(ptr); 
        }
    };
    std::unique_ptr<pcre2_match_data, PCRE2MatchDataDeleter> _match_data;

    // helpers for the initializer
    void init_vocab(const std::vector<std::string>& tokens);
    void init_merge_rules(const std::vector<std::string>& merges);
    void compile_regex_pattern();
    
    // misc helpers (encoding, decoding)
    std::string decode_token_bytes(const std::string& token);
    std::vector<std::string> regex_split(const std::string& text);
    std::vector<uint32_t> bpe_encode(const std::string& token);
};