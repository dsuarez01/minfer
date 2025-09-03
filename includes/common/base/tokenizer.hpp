#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

class BaseTokenizer {
public:

    BaseTokenizer(uint32_t eos_id, uint32_t pad_id) : _eos_id(eos_id), _pad_id(pad_id) {};

    // for chat template
    struct Message {
        std::string role;                   // e.g. "system", "user", "assistant", "tool"
        std::string content;               // main content of the message
        std::string reasoning_content;       // for <think> tags
        std::vector<std::pair<std::string, std::string>> tool_calls;  // name, arguments pairs
        
        Message(const std::string& r, const std::string& c): role(r), content(c) {}
    };

    struct Tool {
        std::string name;
        std::string description;
        std::string parameters;  // JSON str
    };

    virtual ~BaseTokenizer() = default;
    
    virtual std::vector<uint32_t> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<uint32_t>& tokens) = 0;
    virtual std::string decode_token(uint32_t token) = 0;
    
    // apply chat template to messages
    virtual std::string apply_chat_template(
        const std::vector<Message>& messages,
        const std::vector<Tool>& tools = {},
        bool add_generation_prompt = true
    ) = 0;

    uint32_t get_eos_id() const { return _eos_id; };
    uint32_t get_pad_id() const { return _pad_id; };

private:
    uint32_t _eos_id;
    uint32_t _pad_id;

};