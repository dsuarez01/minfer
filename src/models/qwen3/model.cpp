#include "minfer/config/config.hpp"
#include "minfer/models/qwen3/model.hpp"
#include "minfer/models/qwen3/module.hpp"
#include "minfer/models/qwen3/tokenizer.hpp"

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

    auto embed_weight = model_data->tensors.at("token_embd.weight");

    auto embed = std::make_unique<Qwen3Embed>(
        config->vocab_size, config->d_model, 
        embed_weight, 
        embed_weight->dtype,
        embed_weight->device
    );

    append_layer(std::move(embed));

    // each of n_layers decoder blocks are split into GQA and MoE
    for (int i = 0; i < config->n_layers; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";
        auto dtype_device_gqa = model_data->tensors.at(layer_prefix + "attn_q.weight"); // this is usually quantized
        auto qdtype_gqa = dtype_device_gqa->dtype;
        auto device_gqa = dtype_device_gqa->device;
        auto gqa = std::make_unique<Qwen3GQA>(
            i, // block_idx
            config->d_model, config->user_max_seq_len,
            config->n_heads, config->n_kv_heads, config->d_head, config->d_rotary,
            config->rms_norm_eps, config->freq_base,
            model_data->tensors.at(layer_prefix + "attn_q.weight"),
            model_data->tensors.at(layer_prefix + "attn_k.weight"),
            model_data->tensors.at(layer_prefix + "attn_v.weight"),
            model_data->tensors.at(layer_prefix + "attn_output.weight"),
            model_data->tensors.at(layer_prefix + "attn_q_norm.weight"),
            model_data->tensors.at(layer_prefix + "attn_k_norm.weight"),
            model_data->tensors.at(layer_prefix + "attn_norm.weight"), 
            qdtype_gqa,
            device_gqa
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

        auto dtype_device_moe = model_data->tensors.at(gate_key); // this is usually quantized
        auto qdtype_moe = dtype_device_moe->dtype;
        auto device_moe =  dtype_device_moe->device;
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
            model_data->tensors.at(up_key),
            qdtype_moe,
            device_moe
        );
        append_layer(std::move(moe));
    }
    
    auto final_norm_weight = model_data->tensors.at("output_norm.weight");
    auto final_norm = std::make_unique<Qwen3FinalRMSNorm>(
        config->d_model, config->rms_norm_eps,
        final_norm_weight,
        final_norm_weight->dtype,
        final_norm_weight->device
    );
    append_layer(std::move(final_norm));
    
    auto lm_head = std::make_unique<Qwen3LMHead>(
        config->d_model,
        config->vocab_size,
        embed_weight,
        nullptr,
        embed_weight->dtype,
        embed_weight->device
    );
    append_layer(std::move(lm_head));
}