#include "common/config/config.hpp"
#include "common/models/qwen3/model.hpp"

#include <optional>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> -i <input> -m <max_len> -s <seed> <mode> -n <num_iters>\n";
    std::cout << "Required:\n";
    std::cout << "  -i, --input <text>     Input text to process\n";
    std::cout << "  -m, --max-len <int>    Maximum sequence length\n";
    std::cout << "  -s, --seed <int>       Random seed\n";
    std::cout << "Mode (required):\n";
    std::cout << "  -t, --thinking         Use thinking mode preset\n";
    std::cout << "  --instruct             Use instruct mode preset\n";
    std::cout << "Optional:\n";
    std::cout << "  -n, --num-iters <int>  Upper bound on generated seq len (-m arg by default)\n ";
    std::cout << "===========";
    std::cout << "  -h, --help             Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filepath = argv[1];
    std::string input_text;
    std::optional<size_t> num_iters;
    std::optional<size_t> max_seq_len;
    std::optional<int> seed;
    bool thinking_mode_set = false;
    bool instruct_mode_set = false;

    // parsing args
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                input_text = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
        }
        else if (arg == "-m" || arg == "--max-len") {
            if (i + 1 < argc) {
                max_seq_len = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
        }
        else if (arg == "-s" || arg == "--seed") {
            if (i + 1 < argc) {
                seed = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                return 1;
            }
        }
        else if (arg == "-t" || arg == "--thinking") {
            thinking_mode_set = true;
        }
        else if (arg == "--instruct") {
            instruct_mode_set = true;
        }
        else if (arg == "-n" || arg == "--num_iters") {
            if (i + 1 < argc) {
                num_iters = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a positive integer value if passed in\n";
                return 1;
            }
        }
        else {
            std::cerr << "Error: Unknown argument " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (input_text.empty()) {
        std::cerr << "Error: Input text is required (use -i or --input)\n";
        return 1;
    }
    if (!max_seq_len.has_value() || max_seq_len.value() <= 0) {
        std::cerr << "Error: Positive max sequence length is required (use -m or --max-len)\n";
        return 1;
    }
    if (!seed.has_value()) {
        std::cerr << "Error: Seed is required (use -s or --seed)\n";
        return 1;
    }
    if (!thinking_mode_set && !instruct_mode_set) {
        std::cerr << "Error: Mode is required (use -t or --instruct)\n";
        return 1;
    }
    if (thinking_mode_set && instruct_mode_set) {
        std::cerr << "Error: Cannot specify both thinking and instruct modes\n";
        return 1;
    }
    if (!num_iters.has_value()) {
        num_iters = max_seq_len; // default to max_seq_len if not passed in
    } else if (num_iters.value() <= 0) {
        std::cerr << "Error: num_iters requires a positive integer value\n";
        return 1;
    } else if (num_iters.value() > max_seq_len.value()) {
        std::cerr << "Error: num_iters cannot exceed max_seq_len\n";
        return 1;
    }
    
    // mode-specific parameters
    float temperature, top_p, min_p, penalty_pres;
    size_t top_k;
    
    if (thinking_mode_set) {
        temperature = 0.6f;
        penalty_pres = 1.5f;
        min_p = 0.0f;
        top_p = 0.95f;
        top_k = 20;
    } else {
        temperature = 0.7f;
        penalty_pres = 1.5f;
        min_p = 0.0f;
        top_p = 0.80f;
        top_k = 20;
    }
    
    std::cout << "Model: " << filepath << "\n";
    std::cout << "Input: " << input_text << "\n";
    std::cout << "Mode: " << (thinking_mode_set ? "thinking" : "instruct") << "\n";
    std::cout << "Max length: " << max_seq_len.value() << "\n";
    std::cout << "Seed: " << seed.value() << "\n";
    std::cout << "Temperature: " << temperature << "\n";
    std::cout << "Top-p: " << top_p << "\n";
    std::cout << "Top-k: " << top_k << "\n";
    std::cout << "Min-p: " << min_p << "\n";
    std::cout << "Presence penalty: " << penalty_pres << "\n\n";
    
    ModelData model_data;
    int result = model_data.from_file(filepath);
    if (result != 0) {
        std::cerr << "Failed to load model data" << std::endl;
        return -1;
    }
    
    RunParams run_params(num_iters.value(), max_seq_len.value(), temperature, top_k, top_p, min_p, penalty_pres, seed.value());
    Qwen3Model test(model_data, run_params);
    test.generate(input_text);
    
    return 0;
}