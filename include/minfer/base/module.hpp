#pragma once

#include "minfer/config/config.hpp"

using TPtr = std::shared_ptr<Tensor>;

class BaseLayer {
public:
    BaseLayer(DataType qdtype, DeviceType device) : _qdtype(qdtype), _device(device) {};
    virtual ~BaseLayer() = default; // need virtual public destructor
    virtual void forward(std::shared_ptr<RunState> run_state) = 0; // now we cannot instantiate BaseLayer
    
    void set_device(DeviceType target_device);
    void set_qdtype(DataType qdtype) { this->_qdtype = qdtype; };
    void set_read_bytes(size_t bytes) { _read_bytes = bytes; };

    DeviceType get_device() const { return this->_device; };
    DataType get_qdtype() const { return this->_qdtype; };
    size_t get_read_bytes() const { return this->_read_bytes; }; // defined as num bytes read from weights per forward pass

protected:
    void append_parameter(TPtr tensor);

private:
    std::vector<TPtr> _parameters;
    DataType _qdtype;  // quantized weight dtype (F32, F16, etc)
    DeviceType _device;
    size_t _read_bytes = 0; // set in init. of derived classes
};

// Base classes with unified forward methods
class Embed : public BaseLayer {
public:
    Embed(
        size_t vocab_size, int d_model, 
        TPtr weight, 
        DataType qdtype, DeviceType device
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
        DataType qdtype, DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    int d_in;
    int d_out;
    TPtr weight;
    TPtr bias;
};

class RMSNorm : public BaseLayer {
public:
    RMSNorm(
        int dim, float eps, 
        TPtr weight, 
        DataType qdtype, DeviceType device
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
        int block_idx, int d_model, size_t max_seq_len, 
        int n_heads, int n_kv_heads, int d_head, int d_rotary,
        float eps, float freq_base,
        TPtr wq, TPtr wk, TPtr wv,
        TPtr wo, TPtr wq_norm, TPtr wk_norm,
        TPtr w_attnnorm,
        DataType qdtype, DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    size_t max_seq_len;
    int block_idx, d_model;
    int n_heads, n_kv_heads, d_head, d_rotary;
    float eps, freq_base;

    std::vector<float> rope_table;
    
    TPtr wq, wk, wv, wo, wq_norm, wk_norm, w_attnnorm;

private:
    static std::vector<float> compute_rope_table(size_t max_seq_len, int d_rotary, float freq_base);
};

class MoE : public BaseLayer {
public:
    MoE(
        int d_model, int d_ff, int n_experts, int n_active_experts, float eps,
        TPtr w_moenorm, TPtr w_router, TPtr ws_gate, TPtr ws_down, TPtr ws_up,
        DataType qdtype, DeviceType device
    );
    virtual void forward(std::shared_ptr<RunState> run_state) = 0;

protected:
    int d_model, d_ff, n_experts, n_active_experts;
    float eps;
    
    TPtr w_moenorm, w_router, ws_gate, ws_down, ws_up;
};