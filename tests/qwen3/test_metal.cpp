#include "minfer/models/qwen3/model.hpp"
#include "minfer/config/config.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_gguf_file>" << std::endl;
        return 1;
    }
    
    std::string gguf_path = argv[1];
    
    // default values for run_params (unused as of now)
    RunParams run_params(
        128,      // num_iters
        4096,     // max_seq_len  
        0.5f,     // temperature
        20,       // top_k
        0.95f,     // top_p
        0.05f,    // min_p
        0.2f,     // penalty_pres
        42        // seed
    );

    Qwen3Model test(gguf_path, run_params);
    
    // test: CPU -> GPU -> CPU
    test.set_device(DeviceType::METAL);
    test.set_device(DeviceType::CPU);
    
    return 0;
}