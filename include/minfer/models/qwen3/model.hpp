#pragma once

#include "minfer/base/model.hpp"
#include "minfer/models/qwen3/module.hpp"

class Qwen3Model : public BaseModel {
private:
    std::shared_ptr<Qwen3Embed> _embed;
    std::vector<std::shared_ptr<Qwen3GQA>> _gqas;
    std::vector<std::shared_ptr<Qwen3MoE>> _moes;
    std::shared_ptr<Qwen3FinalRMSNorm> _final_norm;
    std::shared_ptr<Qwen3LMHead> _lm_head;

public:
    Qwen3Model(const std::string& model_file, const RunParams& run_params);
    void forward(std::shared_ptr<RunState> run_state) override;
};