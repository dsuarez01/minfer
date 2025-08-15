#include "gguf.hpp"
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

std::string tensor_type_to_str(TensorType t_type) {
    switch (t_type) {
        case TensorType::F32:     return "F32";
        case TensorType::F16:     return "F16";
        case TensorType::Q4_0:    return "Q4_0";
        case TensorType::Q4_1:    return "Q4_1";
        case TensorType::Q5_0:    return "Q5_0";
        case TensorType::Q5_1:    return "Q5_1";
        case TensorType::Q8_0:    return "Q8_0";
        case TensorType::Q8_1:    return "Q8_1";
        case TensorType::Q2_K:    return "Q2_K";
        case TensorType::Q3_K:    return "Q3_K";
        case TensorType::Q4_K:    return "Q4_K";
        case TensorType::Q5_K:    return "Q5_K";
        case TensorType::Q6_K:    return "Q6_K";
        case TensorType::Q8_K:    return "Q8_K";
        case TensorType::IQ2_XXS: return "IQ2_XXS";
        case TensorType::IQ2_XS:  return "IQ2_XS";
        case TensorType::IQ3_XXS: return "IQ3_XXS";
        case TensorType::IQ1_S:   return "IQ1_S";
        case TensorType::IQ4_NL:  return "IQ4_NL";
        case TensorType::IQ3_S:   return "IQ3_S";
        case TensorType::IQ2_S:   return "IQ2_S";
        case TensorType::IQ4_XS:  return "IQ4_XS";
        case TensorType::I8:      return "I8";
        case TensorType::I16:     return "I16";
        case TensorType::I32:     return "I32";
        case TensorType::I64:     return "I64";
        case TensorType::F64:     return "F64";
        case TensorType::IQ1_M:   return "IQ1_M";
        case TensorType::BF16:    return "BF16";
        case TensorType::TQ1_0:   return "TQ1_0";
        case TensorType::TQ2_0:   return "TQ2_0";
        case TensorType::MXFP4:   return "MXFP4";
        case TensorType::COUNT:   return "COUNT";
        default:                  return "UNKNOWN";
    }
}

bool is_valid_gguf_key(const std::string& key) {
    return key.length() <= 65535 && 
           std::regex_match(key, std::regex("^[a-z0-9_]+(?:\\.[a-z0-9_]+)*$"));
}

template <typename T>
T read_value(uint8_t*& ptr) {
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

GGUFString read_string(uint8_t*& ptr) {
    GGUFString str;
    str.len = read_value<uint64_t>(ptr);
    str.string = std::string(reinterpret_cast<const char*>(ptr), str.len);
    ptr += str.len;
    return str;
}

GGUFArray read_array(uint8_t*& ptr) {
    GGUFArray arr;
    arr.type = read_value<ValueType>(ptr);
    arr.len = read_value<uint64_t>(ptr);
    arr.array.reserve(arr.len);

    for (uint64_t i = 0; i < arr.len; ++i) {
        arr.array.push_back(read_metadata_value(ptr, arr.type));
    }
    return arr;
}

MetadataValue read_metadata_value(uint8_t*& ptr, ValueType type) {
    switch (type) {
        case ValueType::UINT8:   return read_value<uint8_t>(ptr);
        case ValueType::INT8:    return read_value<int8_t>(ptr);
        case ValueType::UINT16:  return read_value<uint16_t>(ptr);
        case ValueType::INT16:   return read_value<int16_t>(ptr);
        case ValueType::UINT32:  return read_value<uint32_t>(ptr);
        case ValueType::INT32:   return read_value<int32_t>(ptr);
        case ValueType::FLOAT32: return read_value<float>(ptr);
        case ValueType::UINT64:  return read_value<uint64_t>(ptr);
        case ValueType::INT64:   return read_value<int64_t>(ptr);
        case ValueType::FLOAT64: return read_value<double>(ptr);
        case ValueType::BOOL:    return static_cast<bool>(read_value<uint8_t>(ptr));
        case ValueType::STRING:  return read_string(ptr);
        case ValueType::ARRAY:   return read_array(ptr);
        default:
            std::cerr << "Unknown ValueType: " << static_cast<uint32_t>(type) << std::endl;
            throw std::runtime_error("Invalid metadata value type");
    }
}

uint64_t align_offset(uint64_t offset, uint64_t alignment) {
    return offset + (alignment - (offset % alignment)) % alignment;
}

int GGUFFile::from_file(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }
    size_t size = st.st_size;
    void* data = mmap(NULL, size, PROT_READ , MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
        close(fd);
        return -1;
    }

    close(fd); // doesn't invalidate mapping

    uint8_t* ptr = static_cast<uint8_t*>(data);
    uint8_t* start = ptr;

    header.magic = read_value<uint32_t>(ptr);
    header.version = read_value<uint32_t>(ptr);
    header.tensor_count = read_value<uint64_t>(ptr);
    header.metadata_kv_count = read_value<uint64_t>(ptr);

    if (header.magic != 0x46554747) {
        std::cerr << "[Bad file] invalid magic number: 0x" << std::hex << header.magic << std::endl;
        munmap(data,size);
        return -1;
    }

    if (header.version != 3) {
        std::cerr << "[Bad file] only GGUF v3 supported, file version number is: " << header.version << std::endl;
        munmap(data,size);
        return -1;
    }

    if (header.tensor_count == 0 || header.metadata_kv_count == 0) {
        std::cerr << "[Bad file] tensor count or metadata KV count is 0. File has tensor_count of " << header.tensor_count << " and metadata_kv_count of " << header.metadata_kv_count << std::endl;
        munmap(data,size);
        return -1;
    }

    header.metadata_kv.reserve(header.metadata_kv_count);
    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        KVPair kv;
        kv.key = read_string(ptr);
        
        if (!is_valid_gguf_key(kv.key.string)) {
            std::cerr << "[Bad file] invalid key format: " << kv.key.string << std::endl;
            munmap(data, size);
            return -1;
        }
        
        kv.value_type = read_value<ValueType>(ptr);
        kv.value = read_metadata_value(ptr, kv.value_type);
        header.metadata_kv.push_back(std::move(kv));
    }

    tensor_infos.reserve(header.tensor_count);
    for (uint64_t i = 0; i < header.tensor_count; ++i) {
        TensorInfo info;
        info.name = read_string(ptr);
        info.n_dimensions = read_value<uint32_t>(ptr);
        info.dimensions.resize(info.n_dimensions);
        for (uint32_t j = 0; j < info.n_dimensions; ++j) {
            info.dimensions[j] = read_value<uint64_t>(ptr);
        }
        info.type = read_value<TensorType>(ptr);
        info.offset = read_value<uint64_t>(ptr);
        tensor_infos.push_back(std::move(info));
    }

    uint32_t alignment = 32; // see spec
    for (const KVPair& kv : header.metadata_kv) {
        if (kv.key.string == "general.alignment" 
            && std::holds_alternative<uint32_t>(kv.value)) {
            alignment = std::get<uint32_t>(kv.value);
            break;
        }
    }

    uint64_t current_pos = ptr - start;
    uint64_t aligned_pos = align_offset(current_pos, alignment);
    tensor_data = start + aligned_pos;
    tensor_data_size = size - static_cast<size_t>(aligned_pos);

    return 0;
}