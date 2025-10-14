#include "minfer/base/config.hpp"
#include "minfer/models/qwen3/model.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [-h] <model_path> [OPTIONS]\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  -s, --seed <int>       Seed for random generation\n";
    std::cout << "  -d, --device <name>    Device to use: 'cpu' or 'metal' (default: cpu)\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  -h, --help             Show this help\n";
}

struct Args {
    std::string filepath;
    int seed = 0;
    DeviceType device = DeviceType::CPU;
    bool help = false;
    bool valid = true;
    std::string error;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            args.help = true;
            return args;
        }
    }
    
    if (argc < 2) {
        args.valid = false;
        args.error = "Model path required";
        return args;
    }
    
    args.filepath = argv[1];
    bool seed_set = false;
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        auto get_next_arg = [&]() -> std::string {
            if (i + 1 >= argc) {
                args.valid = false;
                args.error = arg + " requires a value";
                return "";
            }
            return argv[++i];
        };
        
        if (arg == "-h" || arg == "--help") {
            args.help = true;
        }
        else if (arg == "-s" || arg == "--seed") {
            std::string val = get_next_arg();
            if (!val.empty()) {
                args.seed = std::stoi(val);
                seed_set = true;
            }
        }
        else if (arg == "-d" || arg == "--device") {
            std::string val = get_next_arg();
            if (!val.empty()) {
                if (val == "cpu" || val == "CPU") {
                    args.device = DeviceType::CPU;
                } else if (val == "metal" || val == "METAL") {
                    args.device = DeviceType::METAL;
                } else {
                    args.valid = false;
                    args.error = "Invalid device: " + val + " (use 'cpu' or 'metal')";
                }
            }
        }
        else {
            args.valid = false;
            args.error = "Unknown argument: " + arg;
            break;
        }
        
        if (!args.valid) break;
    }
    
    if (args.valid && !args.help && !seed_set) {
        args.valid = false;
        args.error = "Seed is required (-s/--seed)";
    }
    
    return args;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (!args.valid) {
        std::cerr << "Error: " << args.error << "\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // 4k max context (benchmark only uses 512 positions max)
    size_t max_seq_len = 4096;
    size_t num_iters = max_seq_len;

    // these have to be set, but are unused
    float temp = 0.0f;
    float penalty_pres = 0.0f;
    float min_p = 0.0f;
    float top_p = 0.0f;
    size_t top_k = 1;
    
    std::cout << "Model: " << args.filepath << std::endl;
    std::cout << "Device: " << device_to_str(args.device) << std::endl;
    std::cout << "Max seq. length: " << max_seq_len << std::endl;
    std::cout << "Seed: " << args.seed << std::endl;
    std::cout << std::endl;
    
    RunParams run_params(num_iters, max_seq_len, top_k, temp, top_p, min_p, penalty_pres, args.seed);
    Qwen3Model test(args.filepath, run_params);
    test.set_device(args.device);
    test.benchmark();

    return 0;
}