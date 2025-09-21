#pragma once

#include "minfer/config/config.hpp"
#include "minfer/base/module.hpp"
#include "minfer/base/tokenizer.hpp"
#include "minfer/base/sampler.hpp"

#include <string>

class BaseModel {
public:
    virtual ~BaseModel() = default;
    
    void generate(std::string& input_text);
    void benchmark();
    void set_device(Device target_device);
    Device get_device() const;
    size_t get_read_bytes() const; // represents bytes read from weights per forward pass

protected:
    std::unique_ptr<ModelData> model_data;
    std::shared_ptr<Config> config;
    std::shared_ptr<RunState> run_state;
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<GenStats> stats;
    BaseModel(const std::string& model_file, const RunParams& run_params);
    void append_layer(std::unique_ptr<BaseLayer> layer);
    virtual void forward(std::shared_ptr<RunState> run_state);

private:
    Device _device = Device::CPU;
    size_t _read_bytes = 0; // represents bytes read from weights per forward pass
    std::vector<std::unique_ptr<BaseLayer>> _layers;
};