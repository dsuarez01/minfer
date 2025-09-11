#include "minfer/models/qwen3/tokenizer.hpp"

#include "extern/nlohmann/json.hpp"
#include "extern/utf8.h"

#include <algorithm>
#include <chrono>

using ordered_json = nlohmann::ordered_json;

Qwen3Tokenizer::Qwen3Tokenizer(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& merges,
    const std::vector<uint32_t>& token_types,
    const std::string& chat_template,
    uint32_t eos_id,
    uint32_t pad_id
) : _token_types(token_types), BaseTokenizer(eos_id, pad_id) {
    
    init_vocab(tokens);
    init_merge_rules(merges);
    compile_regex_pattern(); // TO-DO: return this instead of storing internally?

    pcre2_match_data* raw_match_data = pcre2_match_data_create_from_pattern(_compiled_pattern.get(), nullptr);
    if (!raw_match_data) {
        assert(false && "Tokenizer: failed to create match data");
    }
    _match_data.reset(raw_match_data);
    
    _chat_template = std::make_unique<minja::chat_template>(
        chat_template,
        "<|im_start|>",
        "<|im_end|>"
    );
}

std::vector<uint32_t> Qwen3Tokenizer::encode(const std::string& text) {
    if (text.empty()) return {};
    
    std::vector<uint32_t> result;
    auto matches = regex_split(text);
    
    for (const auto& match : matches) {
        if (_special_tokens.count(match)) {
            auto it = _vocab_to_id.find(match);
            result.push_back(it != _vocab_to_id.end() ? it->second : 0);
        } else {
            auto tokens = bpe_encode(match);
            result.insert(result.end(), tokens.begin(), tokens.end());
        }
    }
    
    return result;
}

std::string Qwen3Tokenizer::decode_token(uint32_t token_id) {
    if (token_id >= _vocab.size()) return "";
    
    // fallback: skip unused tokens
    if (_token_types[token_id] == UNUSED_TOKEN) {
        return "";
    }

    return _vocab[token_id];
}

std::string Qwen3Tokenizer::decode(const std::vector<uint32_t>& tokens) {
    std::string result;
    result.reserve(tokens.size() * 4);
    
    for (uint32_t token_id : tokens) {
        if (token_id < _vocab.size() && _token_types[token_id] != UNUSED_TOKEN) {
            result += _vocab[token_id];
        }
    }

    return result;
}

// process vocab, undo "data gym" mapping via codepoints
void Qwen3Tokenizer::init_vocab(const std::vector<std::string>& tokens) {
    _vocab.reserve(tokens.size());
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t token_type = _token_types[i];
        
        std::string processed_token;
        if (token_type == CONTROL_TOKEN || token_type == USER_DEFINED_TOKEN || token_type == UNUSED_TOKEN) {
            processed_token = tokens[i];
            if (token_type == CONTROL_TOKEN || token_type == USER_DEFINED_TOKEN) {
                _special_tokens.insert(tokens[i]);
            }
        } else {
            processed_token = decode_token_bytes(tokens[i]); // convert regular and byte tokens
        }
        
        _vocab.push_back(processed_token);
        _vocab_to_id[processed_token] = i;
    }
}

// preprocess merge rules into vector of MergeRule structs
// undo "data gym" mapping of l,r bytes via codepoints
void Qwen3Tokenizer::init_merge_rules(const std::vector<std::string>& merges) {
    _merge_rules.reserve(merges.size());
    
    for (const auto& merge_rule : merges) {
        size_t space_pos = merge_rule.find(' ');
        if (space_pos == std::string::npos) continue;
        
        std::string left = merge_rule.substr(0, space_pos);
        std::string right = merge_rule.substr(space_pos + 1);
        
        std::string left_decoded = decode_token_bytes(left);
        std::string right_decoded = decode_token_bytes(right);
        std::string merged = left_decoded + right_decoded;
        
        _merge_rules.push_back({left_decoded, right_decoded, merged});
    }
}

// does actual decoding, see llama.cpp repo for more info
std::string Qwen3Tokenizer::decode_token_bytes(const std::string& token) {
    if (token.empty()) return token;
    
    std::vector<uint32_t> codepoints;
    utf8::utf8to32(token.begin(), token.end(), std::back_inserter(codepoints));
    
    std::string result;
    for (uint32_t codep : codepoints) {
        // codepoint = byte value
        if ((codep >= 33 && codep <= 126) ||      // ASCII printable
            (codep >= 161 && codep <= 172) ||     // Latin-1 Â¡ to Â¬
            (codep >= 174 && codep <= 255)) {     // Latin-1 Â® to Ã¿
            result += static_cast<unsigned char>(codep);
        }
        // map codeps back to orig bytes
        else if (codep >= 256) {
            int offset = codep - 256;
            unsigned char original_byte;
            if (offset <= 32) {
                original_byte = offset; // bytes 0-32
            } else if (offset <= 66) {
                original_byte = 127 + (offset - 33); // bytes 127-160
            } else {
                original_byte = 173; // byte 173
            }
            result += original_byte;
        }
        // fallback, convert back to UTF-8
        else {
            std::string utf8_char;
            utf8::utf32to8(&codep, &codep + 1, std::back_inserter(utf8_char));
            result += utf8_char;
        }
    }
    
    return result;
}

void Qwen3Tokenizer::compile_regex_pattern() {
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
        assert(false && "Tokenizer: regex pattern failed to compile.");
    }
    
    _compiled_pattern.reset(raw_pattern);
}

// see PCRE2 documentation, yes i know it's not ideal
std::vector<std::string> Qwen3Tokenizer::regex_split(const std::string& text) {
    std::vector<std::string> matches;
    matches.reserve(text.length() / 10);
    
    size_t pos = 0;
    while (pos < text.length()) {
        // finding next special token from cur pos
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
        
        // otherwise, try regex match only up to next special token
        int result = pcre2_match(
            _compiled_pattern.get(),
            reinterpret_cast<PCRE2_SPTR>(text.c_str()),
            next_special,
            pos,
            0,
            _match_data.get(),
            nullptr
        );
        
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
    
    // BPE: start with individual bytes as strings
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
            result.push_back(it != _vocab_to_id.end() ? it->second : 0); // 0 as fallback
        }
        return result;
    }
    
    // apply merge rules in order
    for (const auto& rule : _merge_rules) {
        if (word.size() <= 1) break;
        
        
        bool found_merge = false;
        std::vector<std::string> new_word;
        
        // does rule apply to word anywhere
        for (size_t i = 0; i < word.size(); ++i) {
            if (i < word.size() - 1 && word[i] == rule.left && word[i+1] == rule.right) {
                if (!found_merge) { // first merge
                    new_word.reserve(word.size());
                    // copy everything we've seen so far
                    new_word.insert(new_word.end(), word.begin(), word.begin() + i);
                    found_merge = true;
                }
                new_word.push_back(rule.merged);
                ++i; // skip next token
            } else if (found_merge) { // second+ merge
                new_word.push_back(word[i]);
            }
        }
        
        if (found_merge) { // replace word with new word if merge found
            word = std::move(new_word);
        }
    }
    
    // convert (possibly) merged word to token IDs
    std::vector<uint32_t> result;
    result.reserve(word.size());
    for (const auto& w : word) {
        auto it = _vocab_to_id.find(w);
        result.push_back(it != _vocab_to_id.end() ? it->second : 0); // 0 as fallback
    }
    
    return result;
}

// applies chat template using Minja, see documentation for more info
// https://github.com/google/minja
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
    
    try {
        minja::chat_template_inputs inputs;
        inputs.messages = messages_json;
        inputs.tools = tools_json;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.extra_context = ordered_json{};
        inputs.now = std::chrono::system_clock::now();
        return _chat_template->apply(inputs, {});
    } catch (const std::exception& e) {
        // fallback to a simple template
        std::string result;
        for (const auto& msg : messages) {
            result += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        }
        if (add_generation_prompt) {
            result += "<|im_start|>assistant\n";
        }
        return result;
    }
}