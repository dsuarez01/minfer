#pragma once

#include "common/config/config.hpp"
#include "common/base/module.hpp"
#include "common/base/tokenizer.hpp"
#include "common/base/sampler.hpp"

#include <string>

class BaseModel {
public:
    virtual ~BaseModel() = default;
    
    void generate(std::string& input_text);
    void set_device(Device target_device);
    Device get_device() const;
    size_t get_size_bytes() const;

protected:
    std::shared_ptr<Config> config;
    std::shared_ptr<RunState> run_state;
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<GenStats> stats;
    BaseModel(const ModelData& model_data, const RunParams& run_params);
    void append_layer(std::shared_ptr<BaseLayer> layer);
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

private:
    Device _device = Device::CPU;
    size_t _size_bytes = 0;
    std::vector<std::shared_ptr<BaseLayer>> _layers;
};