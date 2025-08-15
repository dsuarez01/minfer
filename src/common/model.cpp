#include "model.hpp"

// === BASE MODEL ===
BaseModel::BaseModel(const ModelData& model_data, const RuntimeParams& run_params)
    : config(model_data, run_params), pool(config) {}

void BaseModel::forward(Pool& pool) {
    for (auto& layer : layers) {
        layer->forward(pool);
    }
}

void BaseModel::append_layer(std::unique_ptr<BaseLayer> layer) {
    layers.push_back(std::move(layer));
}

void BaseModel::to(Device target_device) {
    if (this->device == target_device) return;
    for (auto& layer : layers) {
        layer->to(device);
    }
    pool.to(device);
    this->device = target_device;
}

Device BaseModel::get_device() const {
    return this->device;
}

// === Qwen3 (and eventually other) MODELS

Qwen3Model::Qwen3Model(const ModelData& model_data, const RuntimeParams& run_params)
    : BaseModel(model_data, run_params) {
    
    // embedding
    append_layer(
        std::make_unique<Qwen3Linear>(
            config.vocab_size, config.embed_dim, model_data.tensors.at("token_embd.weight")
        )
    );

    // rope
    append_layer(
        std::make_unique<Qwen3RoPE>(
            config.head_dim, config.user_max_seq_len, config.theta
        )
    );

    // each of n_layers decoder blocks
    for (uint64_t i = 0; i < config.n_layers; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";
        
        DBTensors layer_tensors{
            model_data.tensors.at(layer_prefix + "attn_q.weight"),
            model_data.tensors.at(layer_prefix + "attn_k.weight"), 
            model_data.tensors.at(layer_prefix + "attn_v.weight"),
            model_data.tensors.at(layer_prefix + "attn_output.weight"),
            model_data.tensors.at(layer_prefix + "attn_q_norm.weight"),
            model_data.tensors.at(layer_prefix + "attn_k_norm.weight"),
            
            // dense FFN tensors (nullptr if MoE)
            config.is_moe ? nullptr : model_data.tensors.at(layer_prefix + "ffn_gate.weight"),
            config.is_moe ? nullptr : model_data.tensors.at(layer_prefix + "ffn_down.weight"), 
            config.is_moe ? nullptr : model_data.tensors.at(layer_prefix + "ffn_up.weight"),
            
            // MoE tensors (nullptr if dense)
            config.is_moe ? model_data.tensors.at(layer_prefix + "ffn_gate_inp.weight") : nullptr,
            config.is_moe ? model_data.tensors.at(layer_prefix + "ffn_gate_exps.weight") : nullptr,
            config.is_moe ? model_data.tensors.at(layer_prefix + "ffn_down_exps.weight") : nullptr,
            config.is_moe ? model_data.tensors.at(layer_prefix + "ffn_up_exps.weight") : nullptr,
            
            model_data.tensors.at(layer_prefix + "attn_norm.weight"),
            model_data.tensors.at(layer_prefix + "ffn_norm.weight")
        };

        append_layer(
            std::make_unique<Qwen3DB>(
                i, config.n_heads, config.n_kv_heads, config.head_dim,
                config.embed_dim, config.ffn_dim, config.rms_norm_eps,
                config.user_max_seq_len, config.theta,
                config.n_experts, config.moe_top_k, config.is_moe,
                layer_tensors
            )
        );
    }
    
    // final RMSnorm
    append_layer(
        std::make_unique<Qwen3RMSNorm>(
            config.embed_dim, config.rms_norm_eps, model_data.tensors.at("output_norm.weight")
        )
    );
    
    // lm_head
    append_layer(
        std::make_unique<Qwen3Linear>(
            config.embed_dim, config.vocab_size, model_data.tensors.at("token_embd.weight")
        )
    );
}

int Qwen3Model::sample_next_token() {
    return 0; // TODO
}

void Qwen3Model::generate(std::vector<uint32_t>& input_ids) {
    return; // TODO
}