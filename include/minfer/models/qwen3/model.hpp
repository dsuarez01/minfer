#pragma once

#include "minfer/base/model.hpp"

#include <string>

// forward decls.
struct RunParams;

class Qwen3Model : public BaseModel {

public:
    Qwen3Model(const std::string& model_file, const RunParams& run_params);
};