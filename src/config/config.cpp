#include "minfer/config/config.hpp"
#include "minfer/parsing/gguf.hpp"

#include <iostream>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <cstdint>
#include <algorithm>

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

json Tensor::to_json() const {
    std::stringstream ss;
    ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(data);
    
    return {
        {"name", name},
        {"shape", shape},
        {"dtype", dtype_to_str(dtype)},
        {"size_bytes", size_bytes},
        {"device", device == Device::CPU ? "CPU" : "CUDA"},
        {"data_ptr", ss.str()}
    };
}

void Tensor::set_device(Device target_device) {
    if (device == target_device) return;
    // Copy/transfer data from current device to target device (depends on what target_device is so 
    // this is the interface to check)
    // TO-DO: implement this
    device = target_device;
}

int ModelData::from_file(const std::string& filename) {
    // TO-DO, make parsing safe
    GGUFFile gguf_file;
    if (gguf_file.from_file(filename) == -1) {
        std::cerr << "GGUFFile init failed" << std::endl;
        return -1;
    }

    for (const KVPair& kv : gguf_file.header.metadata_kv) {
        convert_metadata_value(kv.key.string, kv.value, metadata);
    }

    for (size_t i=0; i < gguf_file.tensor_infos.size(); ++i) {
        const TensorInfo& info = gguf_file.tensor_infos[i];
        auto tensor = std::make_shared<Tensor>();
        tensor->name = info.name.string;

        DataType dtype = tensor_to_data_type(info.type);
        if (dtype == DataType::INVALID) {
            // TO-DO: should probably clean up GGUF file here
            std::cerr << "[Bad file] unsupported tensor type: " << tensor_type_to_str(info.type) << std::endl;
            return -1;
        }
        tensor->dtype = dtype;
        
        if (info.dimensions.size() == 0 || info.dimensions.size() > 4) {
            // TO-DO: should probably clean up GGUF file here
            std::cerr << "[Bad file] tensor " << tensor->name << " has 0 or > 4 dimensions" << std::endl;
            return -1;
        }

        size_t element_count = 1;
        for (uint64_t dim : info.dimensions) { // uint64_t -> see GGUF spec
            if (dim == 0) {
                // TO-DO: should probably clean up GGUF file here
                std::cerr << "[Bad file] tensor " << tensor->name << " has value 0 in one of the dimensions" << std::endl;
                return -1;
            }
            element_count *= dim;
        }
        tensor->size_bytes = element_count * dtype_size(tensor->dtype);

        // GGUF dimensions vs actual logical layout: see discussion at https://github.com/ggml-org/llama.cpp/issues/6040
        // thus we shape e.g. GGUF dimension [3,4,5] -> [1,5,4,3] our shape
        tensor->shape.fill(1);
        size_t start_idx = 4 - info.dimensions.size();
        for (size_t j=0; j < info.dimensions.size(); ++j) {
            tensor->shape[start_idx + j] = info.dimensions[info.dimensions.size() - 1 - j];
        }

        if (info.offset + tensor->size_bytes > gguf_file.tensor_data_size) {
            // TO-DO: should probably clean up GGUF file here
            std::cerr << "[Bad file] tensor " << tensor->name << " extends beyond file bounds" << std::endl;
            return -1;
        }

        tensor->data = gguf_file.tensor_data + info.offset;
        tensor->device = Device::CPU;
        tensors[tensor->name] = tensor;
        
        tensor_metadata[tensor->name] = tensor->to_json();
    }
    return 0;
}

RunParams::RunParams(
    size_t num_iters,
    size_t max_seq_len, 
    float temperature, 
    size_t top_k, 
    float top_p,
    float min_p,
    float penalty_pres,
    int seed
) : num_iters(num_iters),
    max_seq_len(max_seq_len), 
    temperature(temperature),
    top_k(top_k),  
    top_p(top_p),
    min_p(min_p),
    penalty_pres(penalty_pres),
    seed(seed) {}

Config::Config(const ModelData& model_data, const RunParams& runtime_params) {
    
    const std::string& arch = model_data.metadata.at("general.architecture").get<std::string>();
    if (ModelSupport::SUPPORTED_ARCHITECTURES.find(arch) == ModelSupport::SUPPORTED_ARCHITECTURES.end()) {
        assert(false && "Unsupported architecture");
    }

    std::string prefix = arch + ".";
    const std::string ggml_prefix = "tokenizer.ggml.";
    const std::string tokenizer_prefix = "tokenizer.";
    
    // required
    model_max_seq_len = model_data.metadata.at(prefix + "context_length").get<uint64_t>();
    d_model = model_data.metadata.at(prefix + "embedding_length").get<uint64_t>();
    n_layers = model_data.metadata.at(prefix + "block_count").get<uint64_t>();
    n_heads = model_data.metadata.at(prefix + "attention.head_count").get<uint64_t>();
    n_kv_heads = model_data.metadata.at(prefix + "attention.head_count_kv").get<uint64_t>();

    int d_k_head = model_data.metadata.at(prefix + "attention.key_length").get<uint64_t>();
    int d_v_head = model_data.metadata.at(prefix + "attention.value_length").get<uint64_t>();
    assert(d_k_head == d_v_head && "d_k_head != d_v_head");
    d_head = d_k_head;

    d_ff = model_data.metadata.contains(prefix + "expert_feed_forward_length") 
       ? model_data.metadata[prefix + "expert_feed_forward_length"].get<uint64_t>()
       : model_data.metadata[prefix + "feed_forward_length"].get<uint64_t>(); // required
    rms_norm_eps = model_data.metadata.at(prefix + "attention.layer_norm_rms_epsilon").get<float>();
    
    // will have to refactor this to be a qwen3-specific config at some point
    d_rotary = d_head;
    freq_base = model_data.metadata.at(prefix + "rope.freq_base").get<float>();
    
    // optional moe, defaults to non-moe values
    n_experts = model_data.metadata.value(prefix + "expert_count", 0);
    n_active_experts = model_data.metadata.value(prefix + "expert_used_count", 0);
    assert(n_active_experts <= n_experts && "n_active_experts > n_experts");

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
    
    num_iters = runtime_params.num_iters;
    user_max_seq_len = runtime_params.max_seq_len;
    assert(user_max_seq_len <= model_max_seq_len && "user supplied max_seq_len > model_max_seq_len (n_ctx)");
    temperature = runtime_params.temperature;
    top_k = runtime_params.top_k;
    top_p = runtime_params.top_p;
    min_p = runtime_params.min_p;
    penalty_pres = runtime_params.penalty_pres;
    seed = runtime_params.seed;
}

RunState::RunState(const std::shared_ptr<Config> config) {
    device = Device::CPU;
    cur_pos = 0;
    token_id = 0;
    compute_logits = false;

    // activations
    x = std::make_unique<float[]>(config->d_model);
    xb = std::make_unique<float[]>(config->d_model);
    xb2 = std::make_unique<float[]>(config->d_model);
    
    // FFN
    hb = std::make_unique<float[]>(config->d_ff);
    hb2 = std::make_unique<float[]>(config->d_ff);
    
    // attn
    q = std::make_unique<float[]>(config->n_heads*config->d_head);
    k = std::make_unique<float[]>(config->n_kv_heads*config->d_head);
    v = std::make_unique<float[]>(config->n_kv_heads*config->d_head);
    att_scores = std::make_unique<float[]>(config->n_heads*config->user_max_seq_len);
    att_out = std::make_unique<float[]>(config->n_heads * config->d_head);
    
    // kv cache
    k_cache = std::make_unique<float[]>(config->n_layers *config->n_kv_heads * config->user_max_seq_len * config->d_head);
    v_cache = std::make_unique<float[]>(config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);
    
    // MoE buffers (reduces to dense case when n_experts, n_active_experts = 0)
    moe_scores = std::make_unique<float[]>(std::max(config->n_experts, 1));
    active_experts = std::make_unique<int[]>(std::max(config->n_active_experts, 1));
    active_experts_scores = std::make_unique<float[]>(std::max(config->n_active_experts, 1));
    active_experts_weights = std::make_unique<float[]>(std::max(config->n_active_experts, 1));

    logits = std::make_unique<float[]>(config->vocab_size);

    // bytes req for bufs
    buffer_bytes = 0;
    buffer_bytes += 3*config->d_model*sizeof(float); // act
    buffer_bytes += 2*config->d_ff*sizeof(float); // FFN
    buffer_bytes += config->n_heads*config->d_head*sizeof(float); // q
    buffer_bytes += 2*config->n_kv_heads*config->d_head*sizeof(float); // k, v
    buffer_bytes += config->n_heads*config->user_max_seq_len*sizeof(float); // att_scores
    buffer_bytes += config->n_heads*config->d_head*sizeof(float); // att_out
    buffer_bytes += std::max(config->n_experts,1)*sizeof(float); // moe_scores
    buffer_bytes += std::max(config->n_active_experts,1)*sizeof(int); // active_experts
    buffer_bytes += 2*std::max(config->n_active_experts,1)*sizeof(float); // active_expert_scores/weights
    buffer_bytes += config->vocab_size*sizeof(float);

    // bytes req for kv cache per position
    kv_bytes_per_pos = 0;
    kv_bytes_per_pos += 2 * config->n_layers * config->n_kv_heads * config->d_head * sizeof(float);
}

void RunState::set_device(Device target_device) {
    if (device == target_device) return;
    
    // TODO: Implement device transfer logic
    // just update device flag for now
    device = target_device;
}

void GenStats::start_timer() {
    timer_start = std::chrono::high_resolution_clock::now();
}

float GenStats::get_elapsed_sec() const {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-timer_start);
    return duration.count() / 1000000.0f;
}

void GenStats::print_stats() const {
    std::cout << "Number of tokens generated: " << this->num_tokens_gen << " toks\n"
              << "Prefill time: " << std::fixed << std::setprecision(3) << this->prefill_time << " sec(s)\n" 
              << "Generation throughput: " << std::setprecision(2) << this->throughput << " tok/sec\n"
              << "Mem. Bandwidth: " << std::setprecision(3) << this->bandwidth << " GB/sec" << std::endl;
}