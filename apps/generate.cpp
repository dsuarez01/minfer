#include "minfer/base/config.hpp"
#include "minfer/models/qwen3/model.hpp"

#include <optional>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [-h] <model_path> [OPTIONS]\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  -d, --device <name>    Device to use: 'cpu' or 'metal' (default: cpu)\n";
    std::cout << "  -p, --prompt <text>    Input prompt text\n";
    std::cout << "  -m, --max-len <int>    Maximum sequence length\n";
    std::cout << "  -s, --seed <int>       Random seed\n";
    std::cout << "  Mode (choose one):\n";
    std::cout << "    -t, --thinking       Use thinking mode preset\n";
    std::cout << "    -i, --instruct       Use instruct mode preset\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  -n, --num-iters <int>  Generation length limit (default: max-len)\n";
    std::cout << "  -h, --help             Show this help\n";
}

struct Args {
    std::string filepath;
    std::string prompt;
    size_t max_seq_len = 0;
    size_t num_iters = 0;
    int seed = 0;
    DeviceType device = DeviceType::CPU;
    bool thinking_mode = false;
    bool instruct_mode = false;
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
        else if (arg == "-p" || arg == "--prompt") {
            args.prompt = get_next_arg();
        }
        else if (arg == "-m" || arg == "--max-len") {
            std::string val = get_next_arg();
            if (!val.empty()) {
                int parsed = std::atoi(val.c_str());
                if (parsed <= 0) {
                    args.valid = false;
                    args.error = "max-len must be positive";
                } else {
                    args.max_seq_len = parsed;
                }
            }
        }
        else if (arg == "-s" || arg == "--seed") {
            std::string val = get_next_arg();
            if (!val.empty()) {
                args.seed = std::atoi(val.c_str());
            }
        }
        else if (arg == "-t" || arg == "--thinking") {
            args.thinking_mode = true;
        }
        else if (arg == "-i" || arg == "--instruct") {
            args.instruct_mode = true;
        }
        else if (arg == "-n" || arg == "--num-iters") {
            std::string val = get_next_arg();
            if (!val.empty()) {
                int parsed = std::atoi(val.c_str());
                if (parsed <= 0) {
                    args.valid = false;
                    args.error = "num-iters must be positive";
                } else {
                    args.num_iters = parsed;
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
    
    // set default num_iters if not specified
    if (args.num_iters == 0) {
        args.num_iters = args.max_seq_len;
    }
    
    return args;
}

bool validate_args(const Args& args) {
    if (args.help) return true;
    
    if (!args.valid) {
        std::cerr << "Error: " << args.error << "\n";
        return false;
    }
    
    if (args.prompt.empty()) {
        std::cerr << "Error: Prompt text is required (use -p or --prompt)\n";
        return false;
    }
    
    if (args.max_seq_len == 0) {
        std::cerr << "Error: Max sequence length is required (use -m or --max-len)\n";
        return false;
    }
    
    if (!args.thinking_mode && !args.instruct_mode) {
        std::cerr << "Error: Mode is required (use -t for thinking or -i for instruct)\n";
        return false;
    }
    
    if (args.thinking_mode && args.instruct_mode) {
        std::cerr << "Error: Cannot specify both thinking and instruct modes\n";
        return false;
    }
    
    if (args.num_iters > args.max_seq_len) {
        std::cerr << "Error: num-iters cannot exceed max-len\n";
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (!validate_args(args)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // mode-specific parameters
    float temperature, top_p, min_p, penalty_pres;
    size_t top_k;
    
    if (args.thinking_mode) {
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
    
    std::cout << "Model: " << args.filepath << std::endl;
    std::cout << "Device: " << device_to_str(args.device) << std::endl;
    std::cout << "Prompt: " << args.prompt << std::endl;
    std::cout << "Mode: " << (args.thinking_mode ? "thinking" : "instruct") << std::endl;
    std::cout << "Max seq. length: " << args.max_seq_len << std::endl;
    std::cout << "Seed: " << args.seed << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << "Top-p: " << top_p << std::endl;
    std::cout << "Top-k: " << top_k << std::endl;
    std::cout << "Min-p: " << min_p << std::endl;
    std::cout << "Presence penalty: " << penalty_pres << std::endl;
    std::cout << std::endl;
    
    RunParams run_params(args.num_iters, args.max_seq_len, temperature, top_k, top_p, min_p, penalty_pres, args.seed);
    Qwen3Model test(args.filepath, run_params);
    test.set_device(args.device);
    test.generate(args.prompt);
    
    return 0;
}