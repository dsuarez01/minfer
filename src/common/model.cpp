#include "model.hpp"

Qwen3Model::Qwen3Model(const ModelData& model_data, const RuntimeParams& run_params)
    : RootModule(), 
      config(model_data, run_params),
      tensors(model_data.tensors),
      pool(config) {
    
    embed_tokens = std::make_unique<Qwen3Linear>(
        *this, config.vocab_size, config.embed_dim, tensors.at("token_embd.weight")
    );
    
    initial_rope = std::make_unique<Qwen3RoPE>(
        *this, config.head_dim, config.user_max_seq_len, config.theta
    );
    
    layers.reserve(config.n_layers);
    for (uint64_t i = 0; i < config.n_layers; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";
        
        DBTensors layer_tensors{
            tensors.at(layer_prefix + "attn_q.weight"),
            tensors.at(layer_prefix + "attn_k.weight"), 
            tensors.at(layer_prefix + "attn_v.weight"),
            tensors.at(layer_prefix + "attn_output.weight"),
            tensors.at(layer_prefix + "attn_q_norm.weight"),
            tensors.at(layer_prefix + "attn_k_norm.weight"),
            
            // dense FFN tensors (nullptr if MoE)
            config.is_moe ? nullptr : &tensors.at(layer_prefix + "ffn_gate.weight"),
            config.is_moe ? nullptr : &tensors.at(layer_prefix + "ffn_down.weight"), 
            config.is_moe ? nullptr : &tensors.at(layer_prefix + "ffn_up.weight"),
            
            // MoE tensors (nullptr if dense)
            config.is_moe ? &tensors.at(layer_prefix + "ffn_gate_inp.weight") : nullptr,
            config.is_moe ? &tensors.at(layer_prefix + "ffn_gate_exps.weight") : nullptr,
            config.is_moe ? &tensors.at(layer_prefix + "ffn_down_exps.weight") : nullptr,
            config.is_moe ? &tensors.at(layer_prefix + "ffn_up_exps.weight") : nullptr,
            
            tensors.at(layer_prefix + "attn_norm.weight"),
            tensors.at(layer_prefix + "ffn_norm.weight")
        };
        
        auto layer = std::make_unique<Qwen3DB>(
            *this, i, config.n_heads, config.n_kv_heads, config.head_dim,
            config.embed_dim, config.ffn_dim, config.rms_norm_eps,
            config.user_max_seq_len, config.theta,
            config.n_experts, config.moe_top_k, config.is_moe,
            layer_tensors
        );
        
        layers.push_back(std::move(layer));
    }
    
    final_norm = std::make_unique<Qwen3RMSNorm>(
        *this, config.embed_dim, config.rms_norm_eps, tensors.at("output_norm.weight")
    );
    
    lm_head = std::make_unique<Qwen3Linear>(
        *this, config.embed_dim, config.vocab_size, tensors.at("token_embd.weight")
    );
}

void Qwen3Model::forward() {
    embed_tokens->forward(pool);
    initial_rope->forward(pool);
    
    for (auto& layer : layers) {
        layer->forward(pool);
    }
    
    final_norm->forward(pool);
    lm_head->forward(pool);
}

void Qwen3Model::to(Device device) {
    RootModule::to(device);
    pool.to(device);
}

int Qwen3Model::sample_next_token() {
    return 0; // TODO
}

void Qwen3Model::generate(std::vector<uint32_t>& input_ids) {
    return; // TODO
}