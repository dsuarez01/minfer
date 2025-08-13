#ifndef MODEL_HPP
#define MODEL_HPP

#include "config.hpp"
#include "module.hpp"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <optional>
#include <cstdint>

class Qwen3Model : public RootModule {
private:
    Qwen3Config config;
    const std::unordered_map<std::string, Tensor>& tensors;
    Pool pool;
    
    std::unique_ptr<Qwen3Linear> embed_tokens;
    std::unique_ptr<Qwen3RoPE> initial_rope;
    std::vector<std::unique_ptr<Qwen3DB>> layers;
    std::unique_ptr<Qwen3RMSNorm> final_norm;
    std::unique_ptr<Qwen3Linear> lm_head;
    
public:
    Qwen3Model(const ModelData& model_data, const RuntimeParams& run_params);
    void forward() override;
    void to(Device device);
    int sample_next_token();
    void generate(std::vector<uint32_t>& input_ids);
};

#endif