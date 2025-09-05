#include "tokenizer/test_tokenizer.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_gguf_file>" << std::endl;
        return 1;
    }
    
    std::string gguf_path = argv[1];
    
    TestTokenizer test("Qwen3 Tokenizer Roundtrip Tests");
    
    if (!test.init_from_gguf(gguf_path)) {
        std::cerr << "Failed to initialize tokenizer from: " << gguf_path << std::endl;
        return 1;
    }
    
    std::cout << "\nRunning tokenizer tests..." << std::endl;
    test.run_all_tests();
    test.print_summary();
    
    return test.all_passed() ? 0 : 1;
}