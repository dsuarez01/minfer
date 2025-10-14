#include "minfer/base/config.hpp"
#include "minfer/models/qwen3/model.hpp"
#include "minfer/models/qwen3/module.hpp"
#include "minfer/models/qwen3/tokenizer.hpp"
#include "minfer/base/tensor.hpp"

#include <memory>

Qwen3Model::Qwen3Model(const std::string& model_file, const RunParams& run_params)
    : BaseModel(model_file, run_params) {
    
    tokenizer = std::make_unique<Qwen3Tokenizer>(
        config->tokens,
        config->merges,
        config->token_type,
        config->chat_template,
        config->eos_token_id,
        config->padding_token_id
    );

    auto embed = std::make_unique<Qwen3Embed>(
        config->vocab_size, config->d_model,
        model_data->tensors.at("token_embd.weight")
    );

    append_layer(std::move(embed));

    // each of n_layers decoder blocks are split into GQA and MoE
    for (int i = 0; i < config->n_layers; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";
        auto gqa = std::make_unique<Qwen3GQA>(
            i, // block_idx
            config->d_model, config->n_heads, config->n_kv_heads, config->d_head, config->d_rotary,
            config->user_max_seq_len,
            config->rms_norm_eps, config->freq_base,
            model_data->tensors.at(layer_prefix + "attn_q.weight"),
            model_data->tensors.at(layer_prefix + "attn_k.weight"),
            model_data->tensors.at(layer_prefix + "attn_v.weight"),
            model_data->tensors.at(layer_prefix + "attn_output.weight"),
            model_data->tensors.at(layer_prefix + "attn_q_norm.weight"),
            model_data->tensors.at(layer_prefix + "attn_k_norm.weight"),
            model_data->tensors.at(layer_prefix + "attn_norm.weight") 
        );
        append_layer(std::move(gqa));
        
        std::string gate_key = config->n_experts > 0 ? 
        layer_prefix + "ffn_gate_exps.weight" : 
        layer_prefix + "ffn_gate.weight";
        
        std::string down_key = config->n_experts > 0 ? 
        layer_prefix + "ffn_down_exps.weight" : 
        layer_prefix + "ffn_down.weight";

        std::string up_key = config->n_experts > 0 ? 
        layer_prefix + "ffn_up_exps.weight" : 
        layer_prefix + "ffn_up.weight";

        auto moe = std::make_unique<Qwen3MoE>(
            config->d_model, config->d_ff,
            config->n_experts, config->n_active_experts,
            config->rms_norm_eps,
            model_data->tensors.at(layer_prefix + "ffn_norm.weight"),
            // router tensor (nullptr if not MoE)
            config->n_experts>0 ? model_data->tensors.at(layer_prefix + "ffn_gate_inp.weight") : nullptr,
            // gate, down, up tensors (name depends on if MoE)
            model_data->tensors.at(gate_key),
            model_data->tensors.at(down_key),
            model_data->tensors.at(up_key)
        );
        append_layer(std::move(moe));
    }
    
    auto final_norm = std::make_unique<Qwen3FinalRMSNorm>(
        config->d_model, config->rms_norm_eps,
        model_data->tensors.at("output_norm.weight")
    );
    append_layer(std::move(final_norm));
    
    auto lm_head = std::make_unique<Qwen3LMHead>(
        config->d_model,
        config->vocab_size,
        model_data->tensors.at("token_embd.weight"),
        nullptr
    );
    append_layer(std::move(lm_head));
}