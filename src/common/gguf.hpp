#ifndef GGUF_HPP
#define GGUF_HPP

#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <regex>

enum class TensorType: uint32_t {
    // removed prefix GGML_TYPE_
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 = 4, support removed
    // Q4_3 = 5, support removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    // Q4_0_4_4 = 31, support removed
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0   = 34,
    TQ2_0   = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    MXFP4   = 39,
    COUNT   = 40,
};

std::string tensor_type_to_str(TensorType t_type);

enum class ValueType: uint32_t {
    // removed prefix GGUF_METADATA_VALUE_TYPE_

    // The value is a 8-bit unsigned integer.
    UINT8 = 0,
    // The value is a 8-bit signed integer.
    INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    FLOAT64 = 12,
};

// A string in GGUF.
struct GGUFString {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    std::string string;
};

using MetadataValue = std::variant<
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    uint64_t,
    int64_t,
    double,
    bool,
    GGUFString,
    struct GGUFArray
>;

struct GGUFArray {
    ValueType type;
    uint64_t len;
    std::vector<MetadataValue> array;
};

MetadataValue read_metadata_value(uint8_t*& ptr, ValueType type);
bool is_valid_gguf_key(const std::string& key);

struct KVPair {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    GGUFString key;
    // The type of the value.
    // Must be one of the `ValueType` values.
    ValueType value_type;
    // The value.
    MetadataValue value;
};

struct GGUFHeader {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    std::vector<KVPair> metadata_kv;
};

uint64_t align_offset(uint64_t offset, uint64_t alignment);

struct TensorInfo {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    GGUFString name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    std::vector<uint64_t> dimensions;
    // The type of the tensor.
    TensorType type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct GGUFFile {
    // The header of the file.
    GGUFHeader header;

    // Tensor infos, which can be used to locate the tensor data.
    std::vector<TensorInfo> tensor_infos;

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    // uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t* tensor_data;
    size_t tensor_data_size;
    int from_file(const std::string& filename);
};
#endif