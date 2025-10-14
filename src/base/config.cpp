#include "minfer/base/config.hpp"
#include "minfer/parsing/gguf.hpp"
#include "minfer/base/tensor.hpp"
#include "minfer/base/types.hpp"
#include "minfer/interfaces/metal_interface.hpp"

#include "extern/nlohmann/json.hpp"


#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <variant>
#include <unistd.h>

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

    void convert_metadata_value(const std::string& base_key, const MetadataValue& value, minfer_json& result) {
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

    void buf_to_metal(std::variant<float*, MTL::Buffer*>& buf, size_t count) {
        auto* cpu_data = std::get<float*>(buf);
        auto* metal_buf = MetalManager::upload(cpu_data, count * sizeof(float));
        buf = metal_buf;
    }

    void buf_to_metal(std::variant<int*, MTL::Buffer*>& buf, size_t count) {
        auto* cpu_data = std::get<int*>(buf);
        auto* metal_buf = MetalManager::upload(cpu_data, count * sizeof(int));
        buf = metal_buf;
    }
    
    void buf_from_metal(std::variant<float*, MTL::Buffer*>& buf) {
        auto* metal_buf = std::get<MTL::Buffer*>(buf);
        void* cpu_data = MetalManager::cpu_ptr(metal_buf);
        MetalManager::release(metal_buf);
        buf = static_cast<float*>(cpu_data);
    }

    void buf_from_metal(std::variant<int*, MTL::Buffer*>& buf) {
        auto* metal_buf = std::get<MTL::Buffer*>(buf);
        void* cpu_data = MetalManager::cpu_ptr(metal_buf);
        MetalManager::release(metal_buf);
        buf = static_cast<int*>(cpu_data);
    }

    template<typename T>
    T* make_aligned(size_t count) {
        size_t size = count * sizeof(T);
        size_t alignment = getpagesize();
        size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
        void* ptr = std::aligned_alloc(alignment, aligned_size);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
}

std::string device_to_str(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return "CPU"; break;
        case DeviceType::METAL: return "Metal"; break;
        default: return ""; break;
    }
}

std::string dtype_kernel_suffix(DataType dtype) {
    switch(dtype) {
        case DataType::F32: return "_f32";
        case DataType::F16: return "_f16";
        case DataType::BF16: return "_bf16";
        default: return "";
    }
}

DataType tensor_to_data_type(TensorType t_type) {
    switch (t_type) {
        case TensorType::F32: return DataType::F32;
        case TensorType::F16: return DataType::F16;
        case TensorType::BF16: return DataType::BF16;
        default: return DataType::INVALID;
    }
}

size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return 4;
        case DataType::F16: return 2;
        case DataType::BF16: return 2;
        default: return 0;
    }
}

std::string dtype_to_str(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "F32";
        case DataType::F16: return "F16";
        case DataType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

DataType str_to_dtype(std::string& dtype_str) {
    if (dtype_str == "F32") {
        return DataType::F32;
    } else if (dtype_str == "F16") {
        return DataType::F16;
    } else if (dtype_str == "BF16") {
        return DataType::BF16;
    } else {
        return DataType::INVALID;
    }
}

int ModelData::from_file(const std::string& filename) {
    if (gguf_file.from_file(filename) == -1) {
        std::cerr << "GGUFFile init failed" << std::endl;
        return -1;
    }

    metadata = std::make_unique<minfer_json>();
    tensor_metadata = std::make_unique<minfer_json>();

    try {
        for (const KVPair& kv : gguf_file.header.metadata_kv) {
            convert_metadata_value(kv.key.string, kv.value, *metadata);
        }

        for (size_t i=0; i < gguf_file.tensor_infos.size(); ++i) {
            const TensorInfo& info = gguf_file.tensor_infos[i];

            DataType dtype = tensor_to_data_type(info.type);
            if (dtype == DataType::INVALID) {
                std::cerr << "[Bad file] unsupported tensor type: " << tensor_type_to_str(info.type) << std::endl;
                return -1;
            }

            if (info.dimensions.size() == 0 || info.dimensions.size() > 4) {
                std::cerr << "[Bad file] tensor " << info.name.string << " has 0 or > 4 dimensions" << std::endl;
                return -1;
            }

            size_t element_count = 1;
            for (uint64_t dim : info.dimensions) {
                if (dim == 0) {
                    std::cerr << "[Bad file] tensor " << info.name.string << " has value 0 in one of the dimensions" << std::endl;
                    return -1;
                }
                element_count = mul(element_count, static_cast<size_t>(dim));
            }
            
            size_t size_bytes = mul(element_count, dtype_size(dtype));

            // GGUF dimensions vs actual logical layout
            std::array<int,4> shape;
            shape.fill(1);
            size_t start_idx = 4 - info.dimensions.size();
            for (size_t j=0; j < info.dimensions.size(); ++j) {
                shape[start_idx + j] = info.dimensions[info.dimensions.size() - 1 - j];
            }

            check(info.offset, gguf_file.tensor_data_size);
            check(size_bytes, gguf_file.tensor_data_size - info.offset);

            std::byte* raw_ptr = gguf_file.tensor_data + info.offset;
            
            auto tensor = std::make_shared<Tensor>();
            tensor->name = info.name.string;
            tensor->shape = shape;
            tensor->size_bytes = size_bytes;
            tensor->dtype = dtype;
            tensor->device = DeviceType::CPU;
            tensor->data = raw_ptr;
            switch (dtype) {
                case DataType::F32: {
                    tensor->cpu_view = fp32_t(raw_ptr);
                    break;
                }
                case DataType::F16: {
                    tensor->cpu_view = fp16_t(raw_ptr);
                    break;
                }
                case DataType::BF16: {
                    tensor->cpu_view = bf16_t(raw_ptr);
                    break;
                }
                default: throw std::logic_error("Unhandled DataType"); break;
            }
            
            tensors[tensor->name] = tensor;
            (*tensor_metadata)[tensor->name] = tensor->to_json();
        }
    } catch (std::exception& e) {
        std::cerr << "[Parse error] " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

ModelData::~ModelData() {}

RunParams::RunParams(
    size_t num_iters,
    size_t max_seq_len, 
    size_t top_k,
    float temp,
    float top_p,
    float min_p,
    float penalty_pres,
    int seed
) : num_iters(num_iters),
    max_seq_len(max_seq_len), 
    top_k(top_k),
    temp(temp),
    top_p(top_p),
    min_p(min_p),
    penalty_pres(penalty_pres),
    seed(seed) {}

Config::Config(const ModelData& model_data, const RunParams& runtime_params) {
    
    minfer_json& metadata = *model_data.metadata;

    const std::string& arch = metadata.at("general.architecture").get<std::string>();
    if (ModelSupport::SUPPORTED_ARCHITECTURES.find(arch) == ModelSupport::SUPPORTED_ARCHITECTURES.end()) {
        assert(false && "Unsupported architecture");
    }

    std::string prefix = arch + ".";
    const std::string ggml_prefix = "tokenizer.ggml.";
    const std::string tokenizer_prefix = "tokenizer.";
    
    // required
    model_max_seq_len = metadata.at(prefix + "context_length").get<uint64_t>();
    d_model = metadata.at(prefix + "embedding_length").get<uint64_t>();
    n_layers = metadata.at(prefix + "block_count").get<uint64_t>();
    n_heads = metadata.at(prefix + "attention.head_count").get<uint64_t>();
    n_kv_heads = metadata.at(prefix + "attention.head_count_kv").get<uint64_t>();

    int d_k_head = metadata.at(prefix + "attention.key_length").get<uint64_t>();
    int d_v_head = metadata.at(prefix + "attention.value_length").get<uint64_t>();
    assert(d_k_head == d_v_head && "d_k_head != d_v_head");
    d_head = d_k_head;

    d_ff = metadata.contains(prefix + "expert_feed_forward_length") 
       ? metadata[prefix + "expert_feed_forward_length"].get<uint64_t>()
       : metadata[prefix + "feed_forward_length"].get<uint64_t>(); // required
    rms_norm_eps = metadata.at(prefix + "attention.layer_norm_rms_epsilon").get<float>();
    
    // will have to refactor this to be a qwen3-specific config at some point
    d_rotary = d_head;
    freq_base = metadata.at(prefix + "rope.freq_base").get<float>();
    
    // optional moe, defaults to non-moe values
    n_experts = metadata.value(prefix + "expert_count", 0);
    n_active_experts = metadata.value(prefix + "expert_used_count", 0);
    assert(n_active_experts <= n_experts && "n_active_experts > n_experts");

    // required
    tokenizer_model = metadata.at(ggml_prefix + "model").get<std::string>();
    tokenizer_pre = metadata.at(ggml_prefix + "pre").get<std::string>();
    tokens = metadata.at(ggml_prefix + "tokens").get<std::vector<std::string>>();
    token_type = metadata.at(ggml_prefix + "token_type").get<std::vector<uint32_t>>();
    merges = metadata.at(ggml_prefix + "merges").get<std::vector<std::string>>();
    eos_token_id = metadata.at(ggml_prefix + "eos_token_id").get<uint32_t>();
    padding_token_id = metadata.at(ggml_prefix + "padding_token_id").get<uint32_t>();
    add_bos_token = metadata.at(ggml_prefix + "add_bos_token").get<bool>();
    chat_template = metadata.at(tokenizer_prefix + "chat_template").get<std::string>();
    vocab_size = tokens.size();
    
    num_iters = runtime_params.num_iters;
    user_max_seq_len = runtime_params.max_seq_len;
    assert(user_max_seq_len <= model_max_seq_len && "user supplied max_seq_len > model_max_seq_len (n_ctx)");
    temp = runtime_params.temp;
    top_k = runtime_params.top_k;
    top_p = runtime_params.top_p;
    min_p = runtime_params.min_p;
    penalty_pres = runtime_params.penalty_pres;
    seed = runtime_params.seed;
}

RunState::RunState(const std::shared_ptr<Config> config) : config(config) {
    device = DeviceType::CPU;
    cur_pos = 0;
    token_id = 0;
    compute_logits = false;

    // activations
    x = make_aligned<float>(config->d_model);
    xb = make_aligned<float>(config->d_model);
    xb2 = make_aligned<float>(config->d_model);

    // FFN
    hb = make_aligned<float>(config->d_ff);
    hb2 = make_aligned<float>(config->d_ff);

    // attn
    q = make_aligned<float>(config->n_heads * config->d_head);
    k = make_aligned<float>(config->n_kv_heads * config->d_head);
    v = make_aligned<float>(config->n_kv_heads * config->d_head);
    att_scores = make_aligned<float>(config->n_heads * config->user_max_seq_len);
    att_out = make_aligned<float>(config->n_heads * config->d_head);

    // KV cache
    k_cache = make_aligned<float>(config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);
    v_cache = make_aligned<float>(config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);

    // MoE
    moe_scores = make_aligned<float>(std::max(config->n_experts, 1));
    active_experts = make_aligned<int>(std::max(config->n_active_experts, 1));
    active_experts_weights = make_aligned<float>(std::max(config->n_active_experts, 1));

    // for non-MoE models
    if (config->n_experts == 0) {
        std::get<int*>(active_experts)[0] = 0;
        std::get<float*>(active_experts_weights)[0] = 1.0f;
    }

    // logits
    logits = make_aligned<float>(config->vocab_size);

    // bytes for KV cache per pos
    kv_bytes_per_pos = 2 * config->n_layers * config->n_kv_heads * config->d_head * sizeof(float);
}

void RunState::set_device(DeviceType target_device) {
    if (device == target_device) return;
    
    switch (device) {
        // CPU -> X
        case DeviceType::CPU: {
            switch (target_device) {
                case DeviceType::METAL: to_metal(); break;
                default: assert(false && "Not supported"); break;
            }
            break;
        }

        // Metal -> X
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

void RunState::to_metal() {
    buf_to_metal(x, config->d_model);
    buf_to_metal(xb, config->d_model);
    buf_to_metal(xb2, config->d_model);
    buf_to_metal(hb, config->d_ff);
    buf_to_metal(hb2, config->d_ff);
    buf_to_metal(q, config->n_heads * config->d_head);
    buf_to_metal(k, config->n_kv_heads * config->d_head);
    buf_to_metal(v, config->n_kv_heads * config->d_head);
    buf_to_metal(att_scores, config->n_heads * config->user_max_seq_len);
    buf_to_metal(att_out, config->n_heads * config->d_head);
    buf_to_metal(k_cache, config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);
    buf_to_metal(v_cache, config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);
    buf_to_metal(moe_scores, std::max(config->n_experts, 1));
    buf_to_metal(active_experts, std::max(config->n_active_experts, 1));
    buf_to_metal(active_experts_weights, std::max(config->n_active_experts, 1));
    buf_to_metal(logits, config->vocab_size);
}

void RunState::from_metal() {
    buf_from_metal(x);
    buf_from_metal(xb);
    buf_from_metal(xb2);
    buf_from_metal(hb);
    buf_from_metal(hb2);
    buf_from_metal(q);
    buf_from_metal(k);
    buf_from_metal(v);
    buf_from_metal(att_scores);
    buf_from_metal(att_out);
    buf_from_metal(k_cache);
    buf_from_metal(v_cache);
    buf_from_metal(moe_scores);
    buf_from_metal(active_experts);
    buf_from_metal(active_experts_weights);
    buf_from_metal(logits);
}

RunState::~RunState() {
    if (device == DeviceType::CPU) {
        std::free(std::get<float*>(x));
        std::free(std::get<float*>(xb));
        std::free(std::get<float*>(xb2));
        std::free(std::get<float*>(hb));
        std::free(std::get<float*>(hb2));
        std::free(std::get<float*>(q));
        std::free(std::get<float*>(k));
        std::free(std::get<float*>(v));
        std::free(std::get<float*>(att_scores));
        std::free(std::get<float*>(att_out));
        std::free(std::get<float*>(k_cache));
        std::free(std::get<float*>(v_cache));
        std::free(std::get<float*>(moe_scores));
        std::free(std::get<int*>(active_experts));
        std::free(std::get<float*>(active_experts_weights));
        std::free(std::get<float*>(logits));
    } else if (device == DeviceType::METAL) {
        MetalManager::release(std::get<MTL::Buffer*>(x));
        MetalManager::release(std::get<MTL::Buffer*>(xb));
        MetalManager::release(std::get<MTL::Buffer*>(xb2));
        MetalManager::release(std::get<MTL::Buffer*>(hb));
        MetalManager::release(std::get<MTL::Buffer*>(hb2));
        MetalManager::release(std::get<MTL::Buffer*>(q));
        MetalManager::release(std::get<MTL::Buffer*>(k));
        MetalManager::release(std::get<MTL::Buffer*>(v));
        MetalManager::release(std::get<MTL::Buffer*>(att_scores));
        MetalManager::release(std::get<MTL::Buffer*>(att_out));
        MetalManager::release(std::get<MTL::Buffer*>(k_cache));
        MetalManager::release(std::get<MTL::Buffer*>(v_cache));
        MetalManager::release(std::get<MTL::Buffer*>(moe_scores));
        MetalManager::release(std::get<MTL::Buffer*>(active_experts));
        MetalManager::release(std::get<MTL::Buffer*>(active_experts_weights));
        MetalManager::release(std::get<MTL::Buffer*>(logits));
    }
}