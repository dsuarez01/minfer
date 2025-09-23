#include "minfer/config/config.hpp"
#include "minfer/config/metal_config.hpp"

#ifdef USE_METAL
    #define NS_PRIVATE_IMPLEMENTATION
    #define CA_PRIVATE_IMPLEMENTATION  
    #define MTL_PRIVATE_IMPLEMENTATION
    #include "Metal/Metal.hpp"

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

        struct MetalContext {
            std::unique_ptr<MTL::Device, MetalDeviceDeleter> device;
            std::unique_ptr<MTL::CommandQueue, MetalCmdQueueDeleter> cmd_queue;
        };
        
        static std::unique_ptr<MetalContext> g_metal_ctx;

        template<typename T>
        void buffer_to_metal(std::unique_ptr<T[], AlignedDeleter>& buffer, size_t count) {
            T* old_ptr = buffer.release();
            void* metal_ptr = MetalManager::upload(old_ptr, count * sizeof(T));
            assert(metal_ptr);
            buffer.reset(static_cast<T*>(metal_ptr));
        }

        template<typename T>
        void buffer_from_metal(std::unique_ptr<T[], AlignedDeleter>& buffer) {
            MTL::Buffer* metal_buf = reinterpret_cast<MTL::Buffer*>(buffer.release());
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
            void* buffer = g_metal_ctx->device->newBuffer(data, size, MTL::ResourceStorageModeShared, nullptr);
            assert(buffer);
            return buffer;
        }

        void release(void* metal_buffer) {
            if (metal_buffer) {
                static_cast<MTL::Buffer*>(metal_buffer)->release();
            }
        }
    }

    void Tensor::to_metal() {
        data = MetalManager::upload(data, size_bytes);
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

#else
    // stub
    namespace MetalManager {
        void init() {}
        void* upload(void* data, size_t size) { return nullptr; }
        void release(void* metal_buffer) {}
    }
#endif