#include "minfer/config/config.hpp"
#include "minfer/config/metal_config.hpp"
#include <iostream>

#ifdef USE_METAL
    #define NS_PRIVATE_IMPLEMENTATION
    #define CA_PRIVATE_IMPLEMENTATION  
    #define MTL_PRIVATE_IMPLEMENTATION
    #include "extern/Metal/Metal.hpp"

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

        struct MetalContext {
            std::unique_ptr<MTL::Device, MetalDeviceDeleter> device;
            std::unique_ptr<MTL::CommandQueue, MetalCmdQueueDeleter> cmd_queue;

            std::unique_ptr<MTL::CommandBuffer, MetalCmdBufferDeleter> current_cmd_buffer = nullptr;
            std::unique_ptr<MTL::ComputeCommandEncoder, MetalEncoderDeleter> current_encoder = nullptr;
        };
        
        // one metal context for entire appn
        static std::unique_ptr<MetalContext> g_metal_ctx;

        template<typename T>
        void buffer_to_metal(std::unique_ptr<T[], AlignedDeleter>& buffer, size_t count) {
            T* old_ptr = buffer.release();
            auto* metal_buf = static_cast<MTL::Buffer*>(MetalManager::upload(old_ptr, count * sizeof(T)));
            assert(metal_buf);
            buffer.reset(reinterpret_cast<T*>(metal_buf));
        }

        template<typename T>
        void buffer_from_metal(std::unique_ptr<T[], AlignedDeleter>& buffer) {
            auto* metal_buf = reinterpret_cast<MTL::Buffer*>(buffer.release());
            T* cpu_ptr = static_cast<T*>(metal_buf->contents());
            MetalManager::release(metal_buf);
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
        }

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

        void begin_frame() {
            if (!g_metal_ctx || !g_metal_ctx->cmd_queue) return;

            if (g_metal_ctx->current_cmd_buffer) {
                end_frame();
            }

            MTL::CommandBuffer* cmd_buffer = g_metal_ctx->cmd_queue->commandBuffer();
            g_metal_ctx->current_cmd_buffer.reset(cmd_buffer);

            MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();
            g_metal_ctx->current_encoder.reset(encoder);
        }

        void* get_compute_encoder() {
            return g_metal_ctx ? g_metal_ctx->current_encoder.get() : nullptr;
        }

        void end_frame() {
            if (!g_metal_ctx) return;

            if (g_metal_ctx->current_encoder) {
                g_metal_ctx->current_encoder->endEncoding();
                g_metal_ctx->current_encoder.reset();
            }

            if (g_metal_ctx->current_cmd_buffer) {
                g_metal_ctx->current_cmd_buffer->commit();
                g_metal_ctx->current_cmd_buffer->waitUntilCompleted();
                g_metal_ctx->current_cmd_buffer.reset();
            }
        }

        void* create_pipeline(const char* kernel_src, const char* fcn_name) {
            if (!g_metal_ctx || !g_metal_ctx->device) return nullptr;

            NS::Error* err = nullptr;

            // create lib from src str
            NS::String* src_str = NS::String::string(kernel_src, NS::ASCIIStringEncoding);

            MTL::Library* lib = g_metal_ctx->device->newLibrary(src_str, nullptr, &err);

            if (!lib) {
                std::cerr << "Failed to compile Metal library";
                if (err) {
                    std::cerr << ": " << err->localizedDescription()->utf8String();
                    err->release();
                }
                std::cerr << std::endl;
                src_str->release();
                return nullptr;
            }

            // get fcn from lib
            NS::String* ns_fcn_name = NS::String::string(fcn_name, NS::ASCIIStringEncoding);
            MTL::Function* fcn = lib->newFunction(ns_fcn_name);

            if (!fcn) {
                std::cerr << "Failed to find function: " << fcn_name << std::endl;
                lib->release();
                src_str->release();
                ns_fcn_name->release();
                return nullptr;
            }

            // create compute pipeline
            MTL::ComputePipelineState* pipeline = g_metal_ctx->device->newComputePipelineState(fcn, &err);
            if (!pipeline) {
                std::cerr << "Failed to create compute pipeline state";
                if (err) {
                    std::cerr << ": " << err->localizedDescription()->utf8String();
                    err->release();
                }
                std::cerr << std::endl;
            }

            lib->release();
            fcn->release();
            src_str->release();
            ns_fcn_name->release();

            return pipeline;
        }

        void dispatch_kernel(void* pipeline, void* encoder, size_t threads_x, size_t threads_y, size_t threads_z) {
            auto* pso = static_cast<MTL::ComputePipelineState*>(pipeline);
            auto* enc = static_cast<MTL::ComputeCommandEncoder*>(encoder);
            
            enc->setComputePipelineState(pso);

            MTL::Size grid_size = MTL::Size(threads_x, threads_y, threads_z);

            NS::UInteger max_threads = pso->maxTotalThreadsPerThreadgroup();
            NS::UInteger threadgp_width = std::min(max_threads, (NS::UInteger)threads_x);

            MTL::Size threadgp_size = MTL::Size(threadgp_width, 1, 1);

            enc->dispatchThreads(grid_size, threadgp_size);
        }

    }

    void Tensor::to_metal() {
        MTL::Buffer* metal_buffer = static_cast<MTL::Buffer*>(MetalManager::upload(data, size_bytes));
        data = static_cast<void*>(metal_buffer);
    }

    void Tensor::from_metal() {
        MTL::Buffer* metal_buf = static_cast<MTL::Buffer*>(data);
        void* cpu_ptr = metal_buf->contents();
        MetalManager::release(data);
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