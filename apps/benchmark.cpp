#include "minfer/config/config.hpp"
#include "minfer/models/qwen3/model.hpp"

#include <optional>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> -m <max_len> -s <seed>\n";
    std::cout << "  -h, --help             Show this help\n";
    std::cout << "Required:\n";
    std::cout << "  -m, --max-len <int>    Maximum sequence length\n";
    std::cout << "  -s, --seed <int>       Seed for random gen.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filepath = argv[1];
    
    size_t max_seq_len;
    int seed;

    // parse args (max_seq_len, seed)
    bool max_len_set = false, seed_set = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--max-len") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << argv[i] << " requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
            max_seq_len = std::stoull(argv[++i]);
            max_len_set = true;
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << argv[i] << " requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
            seed = std::stoi(argv[++i]);
            seed_set = true;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!max_len_set || !seed_set) {
        std::cerr << "Error: Both -m/--max-len and -s/--seed are required\n";
        print_usage(argv[0]);
        return 1;
    }

    // run parameters set to default values
    size_t num_iters;
    float temperature, top_p, min_p, penalty_pres;
    size_t top_k;
    
    // default values,
    temperature = 0.0f;
    penalty_pres = 0.0f;
    min_p = 0.0f;
    top_p = 0.0f;
    top_k = 1;
    
    std::cout << "Model: " << filepath << "\n";
    std::cout << "Max seq. length: " << max_seq_len << "\n";
    std::cout << "Seed: " << seed << "\n";
    
    RunParams run_params(num_iters, max_seq_len, temperature, top_k, top_p, min_p, penalty_pres, seed);
    Qwen3Model test(filepath, run_params);
    test.benchmark();

    return 0;
}