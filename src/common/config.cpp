#include "config.hpp"
#include "gguf.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <optional>

// Internal helper functions
namespace {
    template<typename T>
    std::vector<T> extract_array(const GGUFArray& arr) {
        std::vector<T> vec;
        vec.reserve(arr.array.size());
        for (const auto& elem : arr.array) {
            if constexpr (std::is_same_v<T, std::string>) {
                vec.push_back(std::get<GGUFString>(elem).string);
            } else {
                vec.push_back(std::get<T>(elem));
            }
        }
        return vec;
    }

    void convert_metadata_value(const std::string& base_key, const MetadataValue& value, json& result) {
        std::visit([&](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, GGUFString>) {
                result[base_key] = val.string;
            } else if constexpr (std::is_same_v<T, GGUFArray>) {
                switch (val.type) {
                    case ValueType::UINT8:   result[base_key] = extract_array<uint8_t>(val); break;
                    case ValueType::INT8:    result[base_key] = extract_array<int8_t>(val); break;
                    case ValueType::UINT32:  result[base_key] = extract_array<uint32_t>(val); break;
                    case ValueType::INT32:   result[base_key] = extract_array<int32_t>(val); break;
                    case ValueType::FLOAT32: result[base_key] = extract_array<float>(val); break;
                    case ValueType::UINT64:  result[base_key] = extract_array<uint64_t>(val); break;
                    case ValueType::INT64:   result[base_key] = extract_array<int64_t>(val); break;
                    case ValueType::FLOAT64: result[base_key] = extract_array<double>(val); break;
                    case ValueType::BOOL:    result[base_key] = extract_array<bool>(val); break;
                    case ValueType::STRING:  result[base_key] = extract_array<std::string>(val); break;
                    case ValueType::ARRAY:
                        for (size_t i = 0; i < val.array.size(); ++i) {
                            convert_metadata_value(base_key + "." + std::to_string(i), val.array[i], result);
                        }
                        break;
                    default:
                        std::cerr << "Unknown array type: " << static_cast<uint32_t>(val.type) << std::endl;
                        break;
                }
            } else {
                result[base_key] = val;
            }
        }, value);
    }
}

uint64_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return 4;
        case DataType::F16: return 2;
        default: return 0;
    }
}

std::string dtype_to_str(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "F32";
        case DataType::F16: return "F16";
        default: return "UNKNOWN";
    }
}

json Tensor::to_json() const {
    std::stringstream ss;
    ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(data);
    
    return {
        {"name", name},
        {"shape", shape},
        {"dtype", dtype_to_str(dtype)},
        {"size", size},
        {"device", device == Device::CPU ? "CPU" : "CUDA"},
        {"data_ptr", ss.str()}
    };
}

std::optional<DataType> tensor_to_data_type(TensorType t_type) {
    switch (t_type) {
        case TensorType::F32: return DataType::F32;
        case TensorType::F16: return DataType::F16;
        default: return std::nullopt;
    }
}

int ModelData::from_file(const std::string& filename) {
    GGUFFile gguf_file;
    if (gguf_file.from_file(filename) == -1) {
        std::cerr << "GGUFFile init failed" << std::endl;
        return -1;
    }

    for (const KVPair& kv : gguf_file.header.metadata_kv) {
        convert_metadata_value(kv.key.string, kv.value, metadata);
    }

    for (const TensorInfo& info : gguf_file.tensor_infos) {
        Tensor tensor;
        tensor.name = info.name.string;

        auto dtype = tensor_to_data_type(info.type);
        if (!dtype) {
            std::cerr << "[Bad file] unsupported tensor type: " << tensor_type_to_str(info.type) << std::endl;
            return -1;
        }
        tensor.dtype = *dtype;
        
        if (info.dimensions.size() == 0 || info.dimensions.size() > 4) {
            std::cerr << "[Bad file] tensor " << tensor.name << " has 0 or > 4 dimensions" << std::endl;
            return -1;
        }

        uint64_t element_count = 1;
        for (uint64_t dim : info.dimensions) {
            if (dim == 0) {
                std::cerr << "[Bad file] tensor " << tensor.name << " has value 0 in one of the dimensions" << std::endl;
                return -1;
            }
            element_count *= dim;
        }
        tensor.size = element_count * dtype_size(tensor.dtype);

        // shape [3,4] -> [1,1,3,4]
        tensor.shape.fill(1);
        size_t start_idx = 4 - info.dimensions.size();
        for (size_t i = 0; i < info.dimensions.size(); ++i) {
            tensor.shape[start_idx + i] = info.dimensions[i];
        }

        if (info.offset + tensor.size > gguf_file.tensor_data_size) {
            std::cerr << "[Bad file] tensor " << tensor.name << " extends beyond file bounds" << std::endl;
            return -1;
        }

        tensor.data = gguf_file.tensor_data + info.offset;
        tensor.device = Device::CPU;
        tensors[tensor.name] = tensor;
        
        tensor_metadata[tensor.name] = tensor.to_json();
    }
    return 0;
}

Qwen3Config::Qwen3Config(const ModelData& model_data, const RuntimeParams& runtime_params) {
    const std::string& arch = model_data.metadata.at("general.architecture").get<std::string>();
    if (ModelSupport::SUPPORTED_ARCHITECTURES.find(arch) == ModelSupport::SUPPORTED_ARCHITECTURES.end()) {
        throw std::invalid_argument("Unsupported architecture: " + arch);
    }
    is_moe = (arch == "qwen3moe");

    std::string prefix = arch + ".";
    const std::string ggml_prefix = "tokenizer.ggml.";
    const std::string tokenizer_prefix = "tokenizer.";
    
    // required
    model_max_seq_len = model_data.metadata.at(prefix + "context_length").get<uint64_t>();
    embed_dim = model_data.metadata.at(prefix + "embedding_length").get<uint64_t>();
    n_layers = model_data.metadata.at(prefix + "block_count").get<uint64_t>();
    n_heads = model_data.metadata.at(prefix + "attention.head_count").get<uint64_t>();
    n_kv_heads = model_data.metadata.at(prefix + "attention.head_count_kv").get<uint64_t>();
    head_dim = embed_dim / n_heads;
    ffn_dim = model_data.metadata.at(prefix + "feed_forward_length").get<uint64_t>();
    rms_norm_eps = model_data.metadata.value(prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
    theta = model_data.metadata.value(prefix + "rope.freq_base", 10000.0f);
    
    // optional moe, defaults to non-moe values
    moe_top_k = model_data.metadata.value(prefix + "expert_used_count", 0u);
    n_experts = model_data.metadata.value(prefix + "expert_count", 0u);
    expert_dim = model_data.metadata.value(prefix + "expert_feed_forward_length", 0u);

    // required
    tokenizer_model = model_data.metadata.at(ggml_prefix + "model").get<std::string>();
    tokenizer_pre = model_data.metadata.at(ggml_prefix + "pre").get<std::string>();
    tokens = model_data.metadata.at(ggml_prefix + "tokens").get<std::vector<std::string>>();
    token_type = model_data.metadata.at(ggml_prefix + "token_type").get<std::vector<uint32_t>>();
    merges = model_data.metadata.at(ggml_prefix + "merges").get<std::vector<std::string>>();
    eos_token_id = model_data.metadata.at(ggml_prefix + "eos_token_id").get<uint32_t>();
    padding_token_id = model_data.metadata.at(ggml_prefix + "padding_token_id").get<uint32_t>();
    add_bos_token = model_data.metadata.at(ggml_prefix + "add_bos_token").get<bool>();
    chat_template = model_data.metadata.at(tokenizer_prefix + "chat_template").get<std::string>();
    
    vocab_size = tokens.size();
    
    user_max_seq_len = runtime_params.max_seq_len;
    temperature = runtime_params.temperature;
    top_k = runtime_params.top_k;
    top_p = runtime_params.top_p;
    seed = runtime_params.seed;
}

Pool::Pool(const Qwen3Config& config) {
    device = Device::CPU;

    // activations
    x = new float[config.embed_dim];
    xb = new float[config.embed_dim];
    xb2 = new float[config.embed_dim];
    
    // FFN - always needed
    hb = new float[config.ffn_dim];
    hb2 = new float[config.ffn_dim];
    
    // attn
    q = new float[config.n_heads * config.head_dim];
    k = new float[config.n_kv_heads * config.head_dim];
    v = new float[config.n_kv_heads * config.head_dim];
    att = new float[config.n_heads * config.user_max_seq_len];
    
    logits = new float[config.vocab_size];
    
    // MoE buffers, nullptr for dense models
    moe_weights = config.is_moe && config.n_experts > 0 ? new float[config.n_experts] : nullptr;
    active_experts = config.is_moe && config.moe_top_k > 0 ? new int[config.moe_top_k] : nullptr;
    active_experts_weights = config.is_moe && config.n_experts > 0 ? new float[config.moe_top_k * config.expert_dim] : nullptr;
}

Pool::~Pool() {
    delete[] x;
    delete[] xb;
    delete[] xb2;
    delete[] hb;
    delete[] hb2;
    delete[] q;
    delete[] k;
    delete[] v;
    delete[] att;
    delete[] logits;
    // fine to delete nullptr
    delete[] moe_weights;
    delete[] active_experts_weights;
    delete[] active_experts;
}

void Pool::to(Device target_device) {
    if (device == target_device) return;
    
    // TODO: Implement device transfer logic
    // just update device flag for now
    device = target_device;
}