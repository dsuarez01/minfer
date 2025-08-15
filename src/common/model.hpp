#ifndef MODEL_HPP
#define MODEL_HPP

#include "config.hpp"
#include "module.hpp"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <optional>
#include <cstdint>

class BaseModel {
private:
    Device device = Device::CPU;
    std::vector<std::unique_ptr<BaseLayer>> layers;

protected:
    Pool pool;
    Config config;
    BaseModel(const ModelData& model_data, const RuntimeParams& run_params);
    void forward(Pool& pool);
    void append_layer(std::unique_ptr<BaseLayer> layer);

public:
    virtual ~BaseModel() = default; // need public virtual destructor
    void to(Device target_device);
    Device get_device() const;
};

class Qwen3Model : public BaseModel {
public:
    Qwen3Model(const ModelData& model_data, const RuntimeParams& run_params);
    int sample_next_token();
    void generate(std::vector<uint32_t>& input_ids);
};

#endif