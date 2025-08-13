#include "common/gguf.hpp"
#include "common/config.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_file>" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    std::cout << "Testing GGUF file: " << filename << std::endl;

    // test: load GGUF file directly
    std::cout << "\n=== Test 1: Raw GGUF Loading ===" << std::endl;
    GGUFFile gguf_file;
    int result = gguf_file.from_file(filename);
    if (result != 0) {
        std::cerr << "Failed to load GGUF file" << std::endl;
        return -1;
    }
    
    std::cout << "GGUF file loaded successfully!" << std::endl;
    std::cout << "Magic: 0x" << std::hex << gguf_file.header.magic << std::dec << std::endl;
    std::cout << "Version: " << gguf_file.header.version << std::endl;
    std::cout << "Tensor count: " << gguf_file.header.tensor_count << std::endl;
    std::cout << "Metadata count: " << gguf_file.header.metadata_kv_count << std::endl;
    std::cout << "Tensor data size: " << gguf_file.tensor_data_size << " bytes" << std::endl;

    // test: load through ModelData
    std::cout << "\n=== Test 2: ModelData Loading ===" << std::endl;
    ModelData model;
    result = model.from_file(filename);
    if (result != 0) {
        std::cerr << "Failed to load model data" << std::endl;
        return -1;
    }
    
    std::cout << "ModelData loaded successfully!" << std::endl;
    std::cout << "Metadata entries: " << model.metadata.size() << std::endl;
    std::cout << "Tensors loaded: " << model.tensors.size() << std::endl;

    // test: display results (json dump)
    std::cout << model.metadata.dump(2) << std::endl;
    std::cout << model.tensor_metadata.dump(2) << std::endl;

    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}