#include "minfer/base/tensor.hpp"
#include "minfer/base/types.hpp"
#include "minfer/base/config.hpp"
#include "minfer/interfaces/metal_interface.hpp"
#include "extern/nlohmann/json.hpp"

#include <sstream>
#include <iomanip>

void* Tensor::cpu_ptr() const {
    if (device == DeviceType::CPU) {
        return std::get<std::byte*>(data);
    } else if (device == DeviceType::METAL) {
        return MetalManager::cpu_ptr(std::get<MTL::Buffer*>(data));
    } else {
        std::runtime_error("Unrecognized device");
        return nullptr;
    }
}

MTL::Buffer* Tensor::metal_buf() const {
    return std::get<MTL::Buffer*>(data);
}

void Tensor::set_device(DeviceType target_device) {
    if (device == target_device) return;
    
    switch (device) {
        case DeviceType::CPU: {
            switch (target_device) {
                case DeviceType::METAL: to_metal(); break;
                default: assert(false && "Not supported"); break;
            }
            break;
        }
        case DeviceType::METAL: {
            switch (target_device) {
                case DeviceType::CPU: from_metal(); break;
                default: assert(false && "Not supported"); break;
            }
            break;
        }
        default: assert(false && "Not supported"); break;
    }
    
    device = target_device;
}

minfer_json Tensor::to_json() const {
    std::stringstream ss;
    ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(cpu_ptr());
    
    return {
        {"name", name},
        {"shape", shape},
        {"dtype", dtype_to_str(dtype)},
        {"size_bytes", size_bytes},
        {"device", device_to_str(device)},
        {"data_ptr", ss.str()}
    };
}

void Tensor::to_metal() {
    auto* cpu_data = std::get<std::byte*>(data);
    auto* metal_buf = MetalManager::upload(cpu_data, size_bytes);
    data = metal_buf;
    
    switch(dtype) { // null out cpu view
        case DataType::F32:
            cpu_view = fp32_t(nullptr);
            break;
        case DataType::F16:
            cpu_view = fp16_t(nullptr);
            break;
        case DataType::BF16:
            cpu_view = bf16_t(nullptr);
            break;
        default:
            throw std::runtime_error("Unrecognized DataType");
            break;
    }
}

void Tensor::from_metal() {
    auto* metal_buf = std::get<MTL::Buffer*>(data);
    void* cpu_data = MetalManager::cpu_ptr(metal_buf);
    MetalManager::release(metal_buf);
    data = static_cast<std::byte*>(cpu_data);

    switch(dtype) { // init new cpu view from cpu_data
        case DataType::F32:
            cpu_view = fp32_t(static_cast<std::byte*>(cpu_data));
            break;
        case DataType::F16:
            cpu_view = fp16_t(static_cast<std::byte*>(cpu_data));
            break;
        case DataType::BF16:
            cpu_view = bf16_t(static_cast<std::byte*>(cpu_data));
            break;
        default:
            throw std::runtime_error("Unrecognized DataType");
            break;
    }
}

Tensor::~Tensor() {
    // TO-DO: CPU data is responsibility of GGUFFile to clean up, x2 check
    if (device == DeviceType::METAL) {
        MetalManager::release(std::get<MTL::Buffer*>(data));
    }
}