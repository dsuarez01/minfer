#include "minfer/models/qwen3/tokenizer.hpp"

#include "extern/minja/chat-template.hpp"
#include "extern/nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cassert>

using ordered_json = nlohmann::ordered_json;

Qwen3Tokenizer::Qwen3Tokenizer(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& merges,
    const std::vector<uint32_t>& token_types,
    const std::string& chat_template,
    uint32_t eos_id,
    uint32_t pad_id
) : BaseTokenizer(eos_id, pad_id),
    _vocab(tokens),
    _token_types(token_types),
    _special_tokens(init_special_tokens(_vocab, token_types)),
    _merge_rules(init_merge_rules(merges)),
    _vocab_to_id(init_vocab_map(_vocab)),
    _compiled_pattern(init_regex_pattern()),
    _match_data(init_match_data(_compiled_pattern.get()))
{
    if (tokens.size() != token_types.size()) {
        throw std::invalid_argument("Token and token_type arrays must have same size");
    }
    
    assert(_vocab.size() == _token_types.size());
    assert(_compiled_pattern != nullptr);
    assert(_match_data != nullptr);
    
    _chat_template = std::make_unique<minja::chat_template>(
        chat_template,
        "<|im_start|>",
        "<|im_end|>"
    );
}

Qwen3Tokenizer::~Qwen3Tokenizer() = default;

std::vector<uint32_t> Qwen3Tokenizer::encode(const std::string& text) {
    if (text.empty()) return {};
    
    std::vector<uint32_t> result;
    auto matches = regex_split(text);
    
    // each special token is in _vocab_to_id
    for (const auto& match : matches) {
        if (_special_tokens.count(match)) {
            result.push_back(_vocab_to_id.at(match));
        } else {
            auto tokens = bpe_encode(match);
            result.insert(result.end(), tokens.begin(), tokens.end());
        }
    }
    
    return result;
}

std::string Qwen3Tokenizer::decode_token(uint32_t token_id) {
    if (token_id >= _vocab.size()) {
        throw std::out_of_range("Token ID exceeds vocab size");
    }
    
    if (_token_types[token_id] == UNUSED_TOKEN) {
        return "<UNK>";
    }

    return _vocab[token_id];
}

std::string Qwen3Tokenizer::decode(const std::vector<uint32_t>& tokens) {
    if (tokens.empty()) return "";
    
    std::string result;
    result.reserve(tokens.size() * 4);
    
    for (uint32_t token_id : tokens) {
        result += decode_token(token_id);
    }

    return result;
}

std::unordered_set<std::string> Qwen3Tokenizer::init_special_tokens(
    const std::vector<std::string>& vocab,
    const std::vector<uint32_t>& token_types
) {
    std::unordered_set<std::string> special_tokens;
    
    for (size_t i=0; i<vocab.size(); ++i) {
        uint32_t token_type = token_types[i];
        if (token_type == CONTROL_TOKEN || token_type == USER_DEFINED_TOKEN) {
            special_tokens.insert(vocab[i]);
        }
    }
    
    return special_tokens;
}

std::unordered_map<std::string, uint32_t> Qwen3Tokenizer::init_vocab_map(
    const std::vector<std::string>& vocab
) {
    std::unordered_map<std::string, uint32_t> vocab_to_id;
    vocab_to_id.reserve(vocab.size());
    
    for (size_t i=0; i<vocab.size(); ++i) {
        vocab_to_id[vocab[i]] = static_cast<uint32_t>(i);
    }
    
    return vocab_to_id;
}

std::vector<Qwen3Tokenizer::MergeRule> Qwen3Tokenizer::init_merge_rules(
    const std::vector<std::string>& merges
) {
    std::vector<MergeRule> merge_rules;
    merge_rules.reserve(merges.size());
    
    for (const auto& merge_rule : merges) {
        size_t space_pos = merge_rule.find(' ');
        if (space_pos == std::string::npos) {
            throw std::invalid_argument("Invalid merge rule format: " + merge_rule);
        }
        
        std::string left = merge_rule.substr(0, space_pos);
        std::string right = merge_rule.substr(space_pos + 1);
        std::string merged = left + right;
        
        merge_rules.push_back({left, right, merged});
    }
    
    return merge_rules;
}

std::unique_ptr<pcre2_code, Qwen3Tokenizer::PCRE2Deleter> Qwen3Tokenizer::init_regex_pattern() {
    // tiktoken (cl100k_base): https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    std::string base_pattern = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)";
    
    int error_code;
    PCRE2_SIZE error_offset;
    
    pcre2_code* raw_pattern = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(base_pattern.c_str()),
        base_pattern.length(),
        PCRE2_UTF | PCRE2_UCP,
        &error_code,
        &error_offset,
        nullptr
    );
    
    if (!raw_pattern) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(error_code, buffer, sizeof(buffer));
        throw std::runtime_error("Failed to compile tokenizer regex pattern: " + 
                                std::string(reinterpret_cast<char*>(buffer)));
    }
    
    return std::unique_ptr<pcre2_code, PCRE2Deleter>(raw_pattern);
}

std::unique_ptr<pcre2_match_data, Qwen3Tokenizer::PCRE2MatchDataDeleter> 
Qwen3Tokenizer::init_match_data(pcre2_code* pattern) {
    pcre2_match_data* raw_match_data = pcre2_match_data_create_from_pattern(pattern, nullptr);
    if (!raw_match_data) {
        throw std::runtime_error("Failed to create PCRE2 match data");
    }
    
    return std::unique_ptr<pcre2_match_data, PCRE2MatchDataDeleter>(raw_match_data);
}

std::vector<std::string> Qwen3Tokenizer::regex_split(const std::string& text) {
    assert(_compiled_pattern != nullptr);
    assert(_match_data != nullptr);
    
    std::vector<std::string> matches;
    matches.reserve(text.length() / 10);
    
    size_t pos = 0;
    while (pos < text.length()) {
        // find next special token from cur pos
        size_t next_special = text.length();
        size_t special_len = 0;
        for (const auto& special : _special_tokens) {
            size_t found = text.find(special, pos);
            if (found < next_special) {
                next_special = found;
                special_len = special.length();
            }
        }
        
        // if special token is at cur pos, add to matches and advance
        if (next_special == pos) {
            matches.push_back(text.substr(pos, special_len));
            pos += special_len;
            continue;
        }
        
        // otherwise, try regex match, but only up to next special token
        int result = pcre2_match(
            _compiled_pattern.get(),
            reinterpret_cast<PCRE2_SPTR>(text.c_str()),
            next_special,
            pos,
            0,
            _match_data.get(),
            nullptr
        );
        
        // fallback if no match, just append from text at cur pos
        if (result < 0) {
            matches.push_back(std::string(1, text[pos]));
            pos++;
        } else {
            PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(_match_data.get());
            matches.emplace_back(text, ovector[0], ovector[1] - ovector[0]);
            pos = ovector[1];
        }
    }
    
    return matches;
}

std::vector<uint32_t> Qwen3Tokenizer::bpe_encode(const std::string& token) {
    if (token.empty()) return {};
    
    // start with bytes as strs
    std::vector<std::string> word;
    word.reserve(token.size());
    for (unsigned char byte : token) {
        word.push_back(std::string(1, byte));
    }
    
    if (word.size() <= 1) {
        std::vector<uint32_t> result;
        result.reserve(word.size());
        for (const auto& w : word) {
            auto it = _vocab_to_id.find(w);
            if (it == _vocab_to_id.end()) {
                throw std::runtime_error("Byte token '" + w + "' not found in vocab to id map");
            }
            result.push_back(it->second);
        }
        return result;
    }
    
    // merge rules applied in order
    for (const auto& rule : _merge_rules) {
        if (word.size() <= 1) break;
        
        bool found_merge = false;
        std::vector<std::string> new_word;
        
        for (size_t i=0; i<word.size(); ++i) {
            if (i < word.size()-1 && word[i] == rule.left && word[i+1] == rule.right) {
                if (!found_merge) {
                    new_word.reserve(word.size());
                    new_word.insert(new_word.end(), word.begin(), word.begin() + i);
                    found_merge = true;
                }
                new_word.push_back(rule.merged);
                ++i; // skip next token
            } else if (found_merge) {
                new_word.push_back(word[i]);
            }
        }
        
        if (found_merge) {
            word = std::move(new_word);
        }
    }
    
    // merged word -> token IDs
    std::vector<uint32_t> result;
    result.reserve(word.size());
    for (const auto& w : word) {
        auto it = _vocab_to_id.find(w);
        if (it == _vocab_to_id.end()) {
            throw std::runtime_error("BPE token '" + w + "' not found in vocab to id map");
        }
        result.push_back(it->second);
    }
    
    return result;
}

std::string Qwen3Tokenizer::apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<Tool>& tools,
    bool add_generation_prompt
) {
    ordered_json messages_json = ordered_json::array();
    for (const auto& msg : messages) {
        ordered_json msg_json;
        msg_json["role"] = msg.role;
        msg_json["content"] = msg.content;
        
        if (!msg.reasoning_content.empty()) {
            msg_json["reasoning_content"] = msg.reasoning_content;
        }
        
        if (!msg.tool_calls.empty()) {
            msg_json["tool_calls"] = ordered_json::array();
            for (const auto& tool_call : msg.tool_calls) {
                ordered_json tc_json;
                tc_json["name"] = tool_call.first;
                tc_json["arguments"] = ordered_json::parse(tool_call.second);
                msg_json["tool_calls"].push_back(tc_json);
            }
        }
        
        messages_json.push_back(msg_json);
    }
    
    ordered_json tools_json = ordered_json::array();
    for (const auto& tool : tools) {
        ordered_json tool_json;
        tool_json["name"] = tool.name;
        tool_json["description"] = tool.description;
        tool_json["parameters"] = ordered_json::parse(tool.parameters);
        tools_json.push_back(tool_json);
    }
    
    minja::chat_template_inputs inputs;
    inputs.messages = messages_json;
    inputs.tools = tools_json;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.extra_context = ordered_json{};
    inputs.now = std::chrono::system_clock::now();
    return _chat_template->apply(inputs, {});
}