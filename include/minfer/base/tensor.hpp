#pragma once

#include <string>
#include <array>
#include <cstddef>
#include <variant>

#include "extern/nlohmann/json_fwd.hpp"
#include "minfer/base/types.hpp"

// forward decls.
using minfer_json = nlohmann::json;
enum class DataType : int;
enum class DeviceType : int;
namespace MTL { class Buffer; }


struct Tensor {
    virtual ~Tensor();
    
    std::variant<std::byte*, MTL::Buffer*> data;
    std::string name;
    std::array<int,4> shape;
    size_t size_bytes;
    DataType dtype;
    DeviceType device;
    
    void set_device(DeviceType target_device);
    minfer_json to_json() const;
    void to_metal();
    void from_metal();

    void* cpu_ptr() const;
    
    MTL::Buffer* metal_buf() const;
    
    std::variant<fp32_t, fp16_t, bf16_t> cpu_view;

    template <DataType dtype>
    auto& cpu_typed_view() {
        if constexpr (dtype == DataType::F32) {
            return std::get<fp32_t>(cpu_view);
        } else if constexpr(dtype == DataType::F16) {
            return std::get<fp16_t>(cpu_view);
        } else if constexpr(dtype == DataType::BF16) {
            return std::get<bf16_t>(cpu_view);
        }
    }
};