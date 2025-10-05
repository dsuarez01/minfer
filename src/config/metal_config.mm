#include "minfer/config/config.hpp"
#include "minfer/config/metal_config.hpp"
#include <iostream>

#ifdef USE_METAL
    #define NS_PRIVATE_IMPLEMENTATION
    #define CA_PRIVATE_IMPLEMENTATION  
    #define MTL_PRIVATE_IMPLEMENTATION
    #include "extern/Metal/Metal.hpp"

    // precompiled lib for kernels
    extern "C" {
        extern unsigned char kernels_metallib[];
        extern unsigned int kernels_metallib_len;
    }

    namespace {

        struct MetalDeviceDeleter {
            void operator()(MTL::Device* device) { 
                if (device) device->release(); 
            }
        };
        
        struct MetalCmdQueueDeleter {
            void operator()(MTL::CommandQueue* queue) { 
                if (queue) queue->release(); 
            }
        };

        struct MetalCmdBufferDeleter {
            void operator()(MTL::CommandBuffer* buffer) {
                if (buffer) buffer->release();
            }
        };
        
        struct MetalEncoderDeleter {
            void operator()(MTL::ComputeCommandEncoder* encoder) {
                if (encoder) encoder->release();
            }
        };

        struct MetalLibDeleter {
            void operator()(MTL::Library* lib) {
                if (lib) lib->release();
            }
        };

        struct PipelineDeleter {
            void operator()(MTL::ComputePipelineState* pipeline) {
                if (pipeline) pipeline->release();
            }
        };

        struct MetalContext {
            std::unique_ptr<MTL::Device, MetalDeviceDeleter> device;
            std::unique_ptr<MTL::CommandQueue, MetalCmdQueueDeleter> cmd_queue;
            std::unique_ptr<MTL::Library, MetalLibDeleter> library;

            std::unique_ptr<MTL::CommandBuffer, MetalCmdBufferDeleter> cur_cmd_buffer = nullptr;
            std::unique_ptr<MTL::ComputeCommandEncoder, MetalEncoderDeleter> cur_encoder = nullptr;
            std::unordered_map<
                std::string,
                std::unique_ptr<MTL::ComputePipelineState, PipelineDeleter>
            > pipelines;
        };
        
        // one metal context for entire appn
        static std::unique_ptr<MetalContext> g_metal_ctx;

        void* upload(void* data, size_t size) {
            if (!g_metal_ctx || !g_metal_ctx->device) return nullptr;
            MTL::Buffer* buffer = g_metal_ctx->device->newBuffer(data, size, MTL::ResourceStorageModeShared, nullptr);
            assert(buffer);
            return buffer;
        }

        void release(void* metal_buffer) {
            if (metal_buffer) {
                static_cast<MTL::Buffer*>(metal_buffer)->release();
            }
        }

        template<typename T>
        void buffer_to_metal(std::unique_ptr<T[], AlignedDeleter>& buffer, size_t count) {
            T* old_ptr = buffer.release();
            auto* metal_buf = static_cast<MTL::Buffer*>(upload(old_ptr, count * sizeof(T)));
            assert(metal_buf);
            buffer.reset(reinterpret_cast<T*>(metal_buf));
        }

        template<typename T>
        void buffer_from_metal(std::unique_ptr<T[], AlignedDeleter>& buffer) {
            auto* metal_buf = reinterpret_cast<MTL::Buffer*>(buffer.release());
            T* cpu_ptr = static_cast<T*>(metal_buf->contents());
            release(metal_buf);
            buffer.reset(cpu_ptr);
        }
    }

    namespace MetalManager {
        void init() {
            g_metal_ctx = std::make_unique<MetalContext>();
            g_metal_ctx->device.reset(MTL::CreateSystemDefaultDevice());
            assert(g_metal_ctx->device.get());
            g_metal_ctx->cmd_queue.reset(g_metal_ctx->device->newCommandQueue());
            assert(g_metal_ctx->cmd_queue.get());

            dispatch_data_t lib_data = dispatch_data_create(
                kernels_metallib, kernels_metallib_len,
                dispatch_get_main_queue(), ^{}
            );

            // load, cache precompiled lib
            NS::Error* err = nullptr;
            MTL::Library* lib = g_metal_ctx->device->newLibrary(lib_data, &err);
            assert(lib && "Failed to load Metal library");
            g_metal_ctx->library.reset(lib);

            NS::Array* fcn_names = lib->functionNames();
            for (size_t i=0; i<fcn_names->count(); ++i) {
                NS::String* name = static_cast<NS::String*>(fcn_names->object(i));
                MTL::Function* fcn = lib->newFunction(name);
                assert(fcn);
                
                MTL::ComputePipelineState* pipeline = g_metal_ctx->device->newComputePipelineState(fcn, &err);
                assert(pipeline);
                
                g_metal_ctx->pipelines[name->utf8String()] = 
                    std::unique_ptr<MTL::ComputePipelineState, PipelineDeleter>(pipeline);
                fcn->release();
            }
        }

        void begin_frame() {
            if (!g_metal_ctx || !g_metal_ctx->cmd_queue) return;

            if (g_metal_ctx->cur_cmd_buffer) {
                end_frame();
            }

            MTL::CommandBuffer* cmd_buffer = g_metal_ctx->cmd_queue->commandBuffer();
            g_metal_ctx->cur_cmd_buffer.reset(cmd_buffer);

            MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();
            g_metal_ctx->cur_encoder.reset(encoder);
        }

        void end_frame() {
            if (!g_metal_ctx) return;

            if (g_metal_ctx->cur_encoder) {
                g_metal_ctx->cur_encoder->endEncoding();
                g_metal_ctx->cur_encoder.reset();
            }

            if (g_metal_ctx->cur_cmd_buffer) {
                g_metal_ctx->cur_cmd_buffer->commit();
                g_metal_ctx->cur_cmd_buffer->waitUntilCompleted();
                g_metal_ctx->cur_cmd_buffer.reset();
            }
        }

        void dispatch1d(
            const char* kernel_name,
            size_t n_thrgps, size_t n_thrs,
            const void* params, size_t params_size,
            void** bufs, size_t buf_cnt
        ) {
            auto* enc = g_metal_ctx->cur_encoder.get();
            auto it = g_metal_ctx->pipelines.find(kernel_name);
            assert(it != g_metal_ctx->pipelines.end());
            auto* pipeline = it->second.get();

            enc->setComputePipelineState(pipeline);

            size_t start_idx = 0; // some kernels don't have a params struct
            if (params && params_size > 0) {
                enc->setBytes(params, params_size, 0);
                start_idx = 1;
            }
            
            for (size_t i=0; i<buf_cnt; ++i) {
                auto* buf = static_cast<MTL::Buffer*>(bufs[i]);
                enc->setBuffer(buf, 0, start_idx+i);
            }
            
            MTL::Size grid_size(n_thrgps, 1, 1);
            MTL::Size group_size(n_thrs, 1, 1);
            enc->dispatchThreadgroups(grid_size, group_size);
        }


        void dispatch2d(
            const char* kernel_name,
            size_t n_thrgps_x, size_t n_thrgps_y, size_t n_thrs,
            const void* params, size_t params_size,
            void** bufs, size_t buf_cnt
        ) {
            auto* enc = g_metal_ctx->cur_encoder.get();
            auto it = g_metal_ctx->pipelines.find(kernel_name);
            assert(it != g_metal_ctx->pipelines.end());
            auto* pipeline = it->second.get();

            enc->setComputePipelineState(pipeline);

            size_t start_idx = 0; // some kernels don't have a params struct
            if (params && params_size > 0) {
                enc->setBytes(params, params_size, 0);
                start_idx = 1;
            }
            
            for (size_t i=0; i<buf_cnt; ++i) {
                auto* buf = static_cast<MTL::Buffer*>(bufs[i]);
                enc->setBuffer(buf, 0, start_idx+i);
            }
            
            MTL::Size grid_size(n_thrgps_x, n_thrgps_y, 1);
            MTL::Size group_size(n_thrs, 1, 1);
            enc->dispatchThreadgroups(grid_size, group_size);
        }

        void* buf_contents(void* buffer_ptr) {
            if (!buffer_ptr) return nullptr;
            auto* metal_buf = static_cast<MTL::Buffer*>(buffer_ptr);
            return metal_buf->contents();
        }

    }

    void Tensor::to_metal() {
        MTL::Buffer* metal_buffer = static_cast<MTL::Buffer*>(upload(data, size_bytes));
        data = static_cast<void*>(metal_buffer);
    }

    void Tensor::from_metal() {
        MTL::Buffer* metal_buf = static_cast<MTL::Buffer*>(data);
        void* cpu_ptr = metal_buf->contents();
        release(data);
        data = cpu_ptr;
    }

    void RunState::to_metal() {
        buffer_to_metal(x, config->d_model);
        buffer_to_metal(xb, config->d_model);
        buffer_to_metal(xb2, config->d_model);
        buffer_to_metal(hb, config->d_ff);
        buffer_to_metal(hb2, config->d_ff);
        buffer_to_metal(q, config->n_heads * config->d_head);
        buffer_to_metal(k, config->n_kv_heads * config->d_head);
        buffer_to_metal(v, config->n_kv_heads * config->d_head);
        buffer_to_metal(att_scores, config->n_heads * config->user_max_seq_len);
        buffer_to_metal(att_out, config->n_heads * config->d_head);
        buffer_to_metal(k_cache, config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);
        buffer_to_metal(v_cache, config->n_layers * config->n_kv_heads * config->user_max_seq_len * config->d_head);
        buffer_to_metal(moe_scores, std::max(config->n_experts, 1));
        buffer_to_metal(active_experts, std::max(config->n_active_experts, 1));
        buffer_to_metal(active_experts_scores, std::max(config->n_active_experts, 1));
        buffer_to_metal(active_experts_weights, std::max(config->n_active_experts, 1));
        buffer_to_metal(logits, config->vocab_size);
    }

    void RunState::from_metal() {
        buffer_from_metal(x);
        buffer_from_metal(xb);
        buffer_from_metal(xb2);
        buffer_from_metal(hb);
        buffer_from_metal(hb2);
        buffer_from_metal(q);
        buffer_from_metal(k);
        buffer_from_metal(v);
        buffer_from_metal(att_scores);
        buffer_from_metal(att_out);
        buffer_from_metal(k_cache);
        buffer_from_metal(v_cache);
        buffer_from_metal(moe_scores);
        buffer_from_metal(active_experts);
        buffer_from_metal(active_experts_scores);
        buffer_from_metal(active_experts_weights);
        buffer_from_metal(logits);
    }
#endif