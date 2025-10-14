#include "minfer/parsing/gguf.hpp"

#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <regex>
#include <cstddef>

namespace {
    // mutually recursive fcns need forward defn
    GGUFString read_str(std::byte*& ptr, std::byte* end, size_t file_size);
    GGUFArray read_arr(std::byte*& ptr, std::byte* end, size_t file_size);
    MetadataValue read_val(std::byte*& ptr, ValueType type, std::byte* end, size_t file_size);

    template<typename T>
    T read(std::byte*& ptr, std::byte* end) {
        if (ptr + sizeof(T) > end) {
            throw std::runtime_error("Read past file end");
        }
        T value;
        std::memcpy(&value, ptr, sizeof(T));
        ptr += sizeof(T);
        return value;
    }

    size_t mul(size_t a, size_t b) {
        if (a != 0 && b > SIZE_MAX / a) {
            throw std::runtime_error("Size overflow");
        }
        return a * b;
    }

    void check(size_t size, size_t max) {
        if (size > max) {
            throw std::runtime_error("Size too large");
        }
    }

    bool is_valid_gguf_key(const std::string& key) {
        // len check
        if (key.length() > 65535) return false;
        
        // ASCII check
        for (char c : key) {
            if (static_cast<unsigned char>(c) > 127) return false;
        }
        
        // regex check: lower_snake_case pattern w dots
        return std::regex_match(key, std::regex("^[a-z0-9_]+(?:\\.[a-z0-9_]+)*$"));
    }

    GGUFString read_str(std::byte*& ptr, std::byte* end, size_t file_size) {
        GGUFString str;
        str.len = read<uint64_t>(ptr, end);
        
        // check: str len reasonable
        check(str.len, file_size);
        
        // check: enough bytes remaining
        if (ptr + str.len > end) {
            throw std::runtime_error("String extends past file end");
        }
        
        str.string = std::string(reinterpret_cast<const char*>(ptr), str.len);
        ptr += str.len;
        return str;
    }

    GGUFArray read_arr(std::byte*& ptr, std::byte* end, size_t file_size) {
        GGUFArray arr;
        arr.type = read<ValueType>(ptr, end);
        arr.len = read<uint64_t>(ptr, end);
        
        // check: array length reasonable
        check(arr.len, file_size);
        
        arr.array.reserve(arr.len);
        for (uint64_t i = 0; i < arr.len; ++i) {
            arr.array.push_back(read_val(ptr, arr.type, end, file_size));
        }
        return arr;
    }

    MetadataValue read_val(std::byte*& ptr, ValueType type, std::byte* end, size_t file_size) {
        switch (type) {
            case ValueType::UINT8:   return read<uint8_t>(ptr, end);
            case ValueType::INT8:    return read<int8_t>(ptr, end);
            case ValueType::UINT16:  return read<uint16_t>(ptr, end);
            case ValueType::INT16:   return read<int16_t>(ptr, end);
            case ValueType::UINT32:  return read<uint32_t>(ptr, end);
            case ValueType::INT32:   return read<int32_t>(ptr, end);
            case ValueType::FLOAT32: return read<float>(ptr, end);
            case ValueType::UINT64:  return read<uint64_t>(ptr, end);
            case ValueType::INT64:   return read<int64_t>(ptr, end);
            case ValueType::FLOAT64: return read<double>(ptr, end);
            case ValueType::BOOL:    return static_cast<bool>(read<std::byte>(ptr, end));
            case ValueType::STRING:  return read_str(ptr, end, file_size);
            case ValueType::ARRAY:   return read_arr(ptr, end, file_size);
            default:
                std::cerr << "Unknown ValueType: " << static_cast<uint32_t>(type) << std::endl;
                throw std::runtime_error("Invalid metadata value type");
        }
    }

    uint64_t align_offset(uint64_t offset, uint64_t alignment) {
        return offset + (alignment - (offset % alignment)) % alignment;
    }
}

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

    // store for destructor cleanup
    mmap_data = data;
    mmap_size = size;

    close(fd); // doesn't invalidate mapping

    std::byte* ptr = static_cast<std::byte*>(data);
    std::byte* start = ptr;
    std::byte* end = ptr + size;

    try {
        header.magic = read<uint32_t>(ptr, end);
        header.version = read<uint32_t>(ptr, end);
        header.tensor_count = read<uint64_t>(ptr, end);
        header.metadata_kv_count = read<uint64_t>(ptr, end);

        if (header.magic != 0x46554747) {
            std::cerr << "[Bad file] invalid magic number: 0x" << std::hex << header.magic << std::endl;
            return -1;
        }

        if (header.version != 3) {
            std::cerr << "[Bad file] only GGUF v3 supported, file version number is: " << header.version << std::endl;
            return -1;
        }

        if (header.tensor_count == 0 || header.metadata_kv_count == 0) {
            std::cerr << "[Bad file] tensor count or metadata KV count is 0. File has tensor_count of " << header.tensor_count << " and metadata_kv_count of " << header.metadata_kv_count << std::endl;
            return -1;
        }

        header.metadata_kv.reserve(header.metadata_kv_count);
        for (uint64_t i=0; i<header.metadata_kv_count; ++i) {
            KVPair kv;
            kv.key = read_str(ptr, end, size);
            
            if (!is_valid_gguf_key(kv.key.string)) {
                std::cerr << "[Bad file] invalid key format: " << kv.key.string << std::endl;
                return -1;
            }
            
            kv.value_type = read<ValueType>(ptr, end);
            kv.value = read_val(ptr, kv.value_type, end, size);
            header.metadata_kv.push_back(std::move(kv));
        }

        tensor_infos.reserve(header.tensor_count);
        for (uint64_t i=0; i<header.tensor_count; ++i) {
            TensorInfo info;
            info.name = read_str(ptr, end, size);
            info.n_dimensions = read<uint32_t>(ptr, end);
            info.dimensions.resize(info.n_dimensions);
            for (uint32_t j=0; j<info.n_dimensions; ++j) {
                info.dimensions[j] = read<uint64_t>(ptr, end);
            }
            info.type = read<TensorType>(ptr, end);
            info.offset = read<uint64_t>(ptr, end);
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

    } catch (const std::exception& e) {
        std::cerr << "[Parse error] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

GGUFFile::~GGUFFile() {
    if (mmap_data) {
        munmap(mmap_data, mmap_size);
    }
}