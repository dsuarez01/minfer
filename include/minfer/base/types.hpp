#pragma once
#include <cstdint>
#include <cstddef>

enum class DataType { F32 = 0, F16 = 1, BF16 = 2, INVALID = -1 };

struct base_t {
    std::byte* data;
    base_t() : data(nullptr) {}
    base_t(std::byte* p) : data(p) {}
};

struct fp32_t : base_t {
    fp32_t() = default;
    fp32_t(std::byte* p) : base_t(p) {}
    float* ptr(size_t elem_offset = 0) const { return reinterpret_cast<float*>(data) + elem_offset; }
};

struct fp16_t : base_t {
    fp16_t() = default;
    fp16_t(std::byte* p) : base_t(p) {}
    uint16_t* ptr(size_t elem_offset = 0) const { return reinterpret_cast<uint16_t*>(data) + elem_offset; }
};

struct bf16_t : base_t {
    bf16_t() = default;
    bf16_t(std::byte* p) : base_t(p) {}
    uint16_t* ptr(size_t elem_offset = 0) const { return reinterpret_cast<uint16_t*>(data) + elem_offset; }
};