#pragma once

#include <string>
#include <vector>

// forward decls.
struct ModelData;
struct Config;
struct RunState;
struct RunParams;
struct GenStats;
enum class DeviceType : int;
class BaseLayer;
class BaseTokenizer;
class Sampler;

class BaseModel {
public:
    virtual ~BaseModel();
    
    void generate(std::string& input_text);
    void benchmark();
    void set_device(DeviceType target_device);
    DeviceType get_device() const;
    size_t get_read_bytes() const;

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
    DeviceType _device;
    size_t _read_bytes;
    std::vector<std::unique_ptr<BaseLayer>> _layers;
};