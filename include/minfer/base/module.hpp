#pragma once

#include <memory>
#include <vector>

// forward decls.
struct Tensor;
struct RunState;
enum class DataType : int;
enum class DeviceType : int;

using TPtr = std::shared_ptr<Tensor>;

class BaseLayer {
public:
    BaseLayer(DeviceType device) : _device(device) {};
    virtual ~BaseLayer() = default;
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;
    
    void set_device(DeviceType target_device);
    void set_read_bytes(size_t bytes) { _read_bytes = bytes; };

    DeviceType get_device() const { return this->_device; };
    size_t get_read_bytes() const { return this->_read_bytes; };

protected:
    void append_parameter(TPtr tensor);

private:
    std::vector<TPtr> _parameters;
    DeviceType _device;
    size_t _read_bytes = 0;
};

class Embed : public BaseLayer {
public:
    Embed(
        size_t vocab_size, int d_model, 
        TPtr weight, 
        DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    size_t vocab_size;
    int d_model;
    TPtr weight;
};

class Linear : public BaseLayer {
public:
    Linear(
        int d_in, int d_out,
        TPtr weight, TPtr bias, 
        DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    int d_in, d_out;
    TPtr weight, bias;
};

class RMSNorm : public BaseLayer {
public:
    RMSNorm(
        int dim, float eps, 
        TPtr weight, 
        DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    int dim;
    float eps;
    TPtr weight;
};

class GQA : public BaseLayer {
public:
    GQA(
        int block_idx, int d_model, int n_heads, int n_kv_heads, int d_head, int d_rotary,
        size_t max_seq_len,
        float eps, float freq_base,
        TPtr wq, TPtr wk, TPtr wv,
        TPtr wo, TPtr wq_norm, TPtr wk_norm,
        TPtr w_attnnorm,
        DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    int block_idx, d_model, n_heads, n_kv_heads, d_head, d_rotary;
    size_t max_seq_len;
    float eps, freq_base;
    
    TPtr wq, wk, wv, wo, wq_norm, wk_norm, w_attnnorm;
};

class MoE : public BaseLayer {
public:
    MoE(
        int d_model, int d_ff, int n_experts, int n_active_experts,
        float eps,
        TPtr w_moenorm, TPtr w_router, TPtr ws_gate, TPtr ws_down, TPtr ws_up,
        DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    int d_model, d_ff, n_experts, n_active_experts;
    float eps;
    TPtr w_moenorm, w_router, ws_gate, ws_down, ws_up;
};