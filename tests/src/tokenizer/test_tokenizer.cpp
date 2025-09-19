#include "tokenizer/test_tokenizer.hpp"
#include "minfer/config/config.hpp"
#include <iostream>

TestTokenizer::TestTokenizer(const std::string& name) : TestBase(name) {}

bool TestTokenizer::init_from_gguf(const std::string& gguf_path) {
    ModelData model_data;
    if (model_data.from_file(gguf_path) != 0) {
        std::cerr << "Failed to load GGUF file: " << gguf_path << std::endl;
        return false;
    }
    
    try {
        auto tokens = model_data.metadata.at("tokenizer.ggml.tokens").get<std::vector<std::string>>();
        auto merges = model_data.metadata.at("tokenizer.ggml.merges").get<std::vector<std::string>>();
        auto token_types = model_data.metadata.at("tokenizer.ggml.token_type").get<std::vector<uint32_t>>();
        auto chat_template = model_data.metadata.at("tokenizer.chat_template").get<std::string>();
        auto eos_id = model_data.metadata.at("tokenizer.ggml.eos_token_id").get<uint32_t>();
        auto pad_id = model_data.metadata.at("tokenizer.ggml.padding_token_id").get<uint32_t>();
        
        tokenizer = std::make_unique<Qwen3Tokenizer>(
            tokens, merges, token_types, chat_template, eos_id, pad_id
        );
        
        std::cout << "Loaded tokenizer data:" << std::endl;
        std::cout << "  - Tokens: " << tokens.size() << std::endl;
        std::cout << "  - Merges: " << merges.size() << std::endl;
        std::cout << "  - Token types: " << token_types.size() << std::endl;
        std::cout << "  - EOS token ID: " << eos_id << std::endl;
        std::cout << "  - PAD token ID: " << pad_id << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error extracting tokenizer data: " << e.what() << std::endl;
        return false;
    }
}

void TestTokenizer::assert_roundtrip(const std::string& input, const std::string& test_name) {
    if (!tokenizer) {
        assert_true(false, "Tokenizer not initialized");
        return;
    }
    
    // encode, then decode the input text
    std::vector<uint32_t> tokens = tokenizer->encode(input);

    std::string decoded_text = tokenizer->decode(tokens);
    
    // check if roundtrip is successful
    bool roundtrip_success = (input == decoded_text);
    
    std::string message = test_name.empty() ? 
        "Roundtrip failed for: '" + input + "'" : 
        test_name + " - Roundtrip failed for: '" + input + "'";
    
    if (!roundtrip_success) {
        message += " -> Got: '" + decoded_text + "'";
    }
    
    assert_true(roundtrip_success, message);
}

void TestTokenizer::test_basic_roundtrip() {
    assert_roundtrip("Hello, world!", "Basic text");
    assert_roundtrip("The quick brown fox jumps over the lazy dog.", "Pangram");
    assert_roundtrip("This is a simple test.", "Simple sentence");
}

void TestTokenizer::test_empty_string() {
    assert_roundtrip("", "Empty string");
}

void TestTokenizer::test_special_tokens() {
    assert_roundtrip("<|im_start|>", "im_start token");
    assert_roundtrip("<|im_end|>", "im_end token");
    assert_roundtrip("<|endoftext|>", "endoftext token");
    assert_roundtrip("<|im_start|>user\nHello<|im_end|>", "Chat format");
}

void TestTokenizer::test_unicode_characters() {
    assert_roundtrip("Hello, 世界!", "Mixed ASCII/Chinese");
    assert_roundtrip("Café naïve résumé", "Accented characters");
    assert_roundtrip("🚀 🌟 ✨", "Emoji");
    assert_roundtrip("Привет мир", "Cyrillic");
    assert_roundtrip("مرحبا بالعالم", "Arabic");
}

void TestTokenizer::test_numbers_and_punctuation() {
    assert_roundtrip("1234567890", "Numbers");
    assert_roundtrip("!@#$%^&*()", "Punctuation");
    assert_roundtrip("3.14159", "Float number");
    assert_roundtrip("user@example.com", "Email address");
    assert_roundtrip("https://example.com", "URL");
}

void TestTokenizer::test_mixed_content() {
    assert_roundtrip("Hello 123 world! 🌍", "Mixed ASCII/numbers/emoji");
    assert_roundtrip("Code: `print('hello')`", "Code snippet");
    assert_roundtrip("Math: 2 + 2 = 4", "Math expression");
    assert_roundtrip("Line1\nLine2\tTabbed", "Whitespace characters");
}

void TestTokenizer::test_paste() {
    assert_roundtrip(
        R"(
        The Unicode Technical Committee (UTC) meeting #184 was held last week, July 22 – 24, in Redmond, Washington, hosted by Microsoft. Here are some highlights.
        Finalizing Unicode 17.0
        The top priority was to finalize technical decisions for Unicode 17.0 in preparation for a September 9 release. Beta feedback and a small number of new proposals were considered, and various decisions affecting Unicode 17.0 were taken. 
        The most significant change from the Unicode 17.0 Beta is the removal of 44 characters, based on feedback requesting more time to review these characters and the associated proposals:
        09FF BENGALI LETTER SANSKRIT BA
        0B53 ORIYA SIGN DOT ABOVE
        0B54 ORIYA SIGN DOUBLE DOT ABOVE
        1FADD APPLE CORE
        40 Chisoi script characters and the Chisoi block at 16D80..16DAF
        These characters have been postponed to Unicode 18.0. With this change, the total number of new characters for Unicode 17.0 will be 4,803, including CJK Extension J and four new scripts.
        Glyph changes were also approved for 21 characters, all of which were encoded in earlier versions.
        Certain character property changes were also approved. These include a change to the Word_Break property for 00B8 CEDILLA to accommodate orthographic usage for SENĆOŦEN, an indigenous language spoken in Western Canada. In relation to identifiers and security, the seven scripts added in Unicode 16.0 (Garay, Gurung Khema, Kirat Rai, Ol Onal, Sunuwar, Todhri, and Tulu-Tigalari) will be classified in UAX #31 as Excluded Scripts (Table 4), which means that these will not be included in the General Security Profile for secure identifiers.
        First characters approved for Unicode 18.0
        The tentative plan for new characters to be added in the next Unicode version is usually decided at the fall UTC meeting. The first approvals for Unicode 18.0, however, were decided last week at UTC #184. These include the 44 characters postponed from Unicode 17.0, mentioned above, as well as u+20CE UAE DIRHAM SIGN and 16 geometric symbols used in the manuscripts of the 17th-century polymath Gottfried Wilhelm Leibniz.
        As typically happens at each UTC meeting, several code points were provisionally assigned for other new characters that will be candidates for future versions. 
        For characters approved for 18 or provisionally assigned for future versions, see https://www.unicode.org/alloc/Pipeline.html#future.
        Text Terminal Working Group progress
        A temporary working group was created at UTC #175 to work on improved support for Unicode text in text-only terminal environments, particularly for scripts requiring advanced layout. Due to changes in availability of key participants early on, progress was hindered, but the working group is now meeting regularly. 
        To scope the project, they will prioritize scripts classified in UAX #31 as Recommended. These include a number of scripts for which examples of fixed-width text have not been readily available, and the working group would welcome contributions from anyone with knowledge of prior art for fixed-width Indic text.
        For complete details from UTC #184, see the draft minutes. 
        About the Unicode Standard
        The world relies on digital communications. The Unicode Standard is one of the building blocks for global digital communications, providing the encoding for more than 155,000 characters used by thousands of languages and scripts throughout the world. 
        Each character—letter, diacritic, symbol, emoji, etc.—is represented by a unique numeric code, and has defined properties data that define how characters behave in several text processing algorithms. 
        With this combination, The Unicode Standard provides the foundation for implementations to support the world's writing systems, enabling billions of people across the globe to seamlessly communicate with one another across platforms and devices. The Standard is also the foundation for the suite of code, libraries, data, and products that the Unicode Consortium delivers for robust language support.
        ----------------------------------------------
        Adopt a Character and Support Unicode’s Mission
        Looking to give that special someone a special something?
        Or maybe something to treat yourself?
        🕉️💗🏎️🐨🔥🚀爱₿♜🍀
        Adopt a character or emoji to give it the attention it deserves, while also supporting Unicode’s mission to ensure everyone can communicate in their own languages across all devices.
        Each adoption includes a digital badge and certificate that you can proudly display!
        Have fun and support a good cause
        You can also donate funds or gift stock
        )"
    );
}

void TestTokenizer::run_all_tests() {
    if (!tokenizer) {
        assert_true(false, "Tokenizer not initialized - call initialize_from_gguf() first");
        return;
    }
    
    test_empty_string();
    test_basic_roundtrip();
    test_special_tokens();
    test_numbers_and_punctuation();
    test_unicode_characters();
    test_mixed_content();
    test_paste();
}