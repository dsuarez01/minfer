#include "minfer/base/config.hpp"
#include "minfer/interfaces/metal_interface.hpp"
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
        struct MetalContext {
            MTL::Device* device = nullptr;
            MTL::CommandQueue* cmd_queue = nullptr;
            MTL::Library* library = nullptr;
            
            MTL::CommandBuffer* cur_cmd_buffer = nullptr;
            MTL::ComputeCommandEncoder* cur_encoder = nullptr;
            std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines;
            
            ~MetalContext() {
                // clean up in rev order of creation
                if (cur_encoder) {
                    cur_encoder->release();
                    cur_encoder = nullptr;
                }
                if (cur_cmd_buffer) {
                    cur_cmd_buffer->release();
                    cur_cmd_buffer = nullptr;
                }
                for (auto& [name, pipeline] : pipelines) {
                    if (pipeline) pipeline->release();
                }
                pipelines.clear();
                if (library) {
                    library->release();
                    library = nullptr;
                }
                if (cmd_queue) {
                    cmd_queue->release();
                    cmd_queue = nullptr;
                }
                if (device) {
                    device->release();
                    device = nullptr;
                }
            }
        };
        
        // one metal context for entire appn.
        static std::unique_ptr<MetalContext> g_metal_ctx;
    }

    namespace MetalManager {

        // resource-related
        MTL::Buffer* upload(void* cpu_data, size_t size) {
            if (!g_metal_ctx || !g_metal_ctx->device) return nullptr;
            MTL::Buffer* buffer = g_metal_ctx->device->newBuffer(cpu_data, size, MTL::ResourceStorageModeShared, nullptr);
            assert(buffer);
            return buffer;
        }

        void release(MTL::Buffer* buf) {
            if (buf) buf->release();
        }

        void* cpu_ptr(MTL::Buffer* buf) {
            return buf ? buf->contents() : nullptr;
        }

        // dispatch-related
        void init() {
            g_metal_ctx = std::make_unique<MetalContext>();
            g_metal_ctx->device = MTL::CreateSystemDefaultDevice();
            assert(g_metal_ctx->device);

            g_metal_ctx->cmd_queue = g_metal_ctx->device->newCommandQueue();
            assert(g_metal_ctx->cmd_queue);

            dispatch_data_t lib_data = dispatch_data_create(
                kernels_metallib, kernels_metallib_len,
                dispatch_get_main_queue(), ^{}
            );

            // load, cache precompiled lib
            NS::Error* err = nullptr;
            g_metal_ctx->library = g_metal_ctx->device->newLibrary(lib_data, &err);
            assert(g_metal_ctx->library && "Failed to load Metal library");

            NS::Array* fcn_names = g_metal_ctx->library->functionNames();
            for (size_t i = 0; i < fcn_names->count(); ++i) {
                NS::String* name = static_cast<NS::String*>(fcn_names->object(i));
                MTL::Function* fcn = g_metal_ctx->library->newFunction(name);
                assert(fcn);
                
                MTL::ComputePipelineState* pipeline = g_metal_ctx->device->newComputePipelineState(fcn, &err);
                assert(pipeline);
                
                g_metal_ctx->pipelines[name->utf8String()] = pipeline;
                fcn->release();
            }
        }

        void begin_frame() {
            if (!g_metal_ctx || !g_metal_ctx->cmd_queue) return;

            if (g_metal_ctx->cur_cmd_buffer) {
                end_frame();
            }

            g_metal_ctx->cur_cmd_buffer = g_metal_ctx->cmd_queue->commandBuffer();
            g_metal_ctx->cur_encoder = g_metal_ctx->cur_cmd_buffer->computeCommandEncoder();
        }

        void end_frame() {
            if (!g_metal_ctx) return;

            if (g_metal_ctx->cur_encoder) {
                g_metal_ctx->cur_encoder->endEncoding();
                g_metal_ctx->cur_encoder->release();
                g_metal_ctx->cur_encoder = nullptr;
            }

            if (g_metal_ctx->cur_cmd_buffer) {
                g_metal_ctx->cur_cmd_buffer->commit();
                g_metal_ctx->cur_cmd_buffer->waitUntilCompleted();
                g_metal_ctx->cur_cmd_buffer->release();
                g_metal_ctx->cur_cmd_buffer = nullptr;
            }
        }

        void dispatch1d(
            const char* kernel_name,
            size_t n_thrgps, size_t n_thrs,
            const void* params, size_t params_size,
            MTL::Buffer** bufs, size_t buf_cnt
        ) {
            auto* enc = g_metal_ctx->cur_encoder;
            auto it = g_metal_ctx->pipelines.find(kernel_name);
            assert(it != g_metal_ctx->pipelines.end());
            auto* pipeline = it->second;

            enc->setComputePipelineState(pipeline);

            size_t start_idx = 0; // some kernels don't have a params struct
            if (params && params_size > 0) {
                enc->setBytes(params, params_size, 0);
                start_idx = 1;
            }
            
            for (size_t i = 0; i < buf_cnt; ++i) {
                auto* buf = static_cast<MTL::Buffer*>(bufs[i]);
                enc->setBuffer(buf, 0, start_idx + i);
            }
            
            MTL::Size grid_size(n_thrgps, 1, 1);
            MTL::Size group_size(n_thrs, 1, 1);
            enc->dispatchThreadgroups(grid_size, group_size);
        }

        void dispatch2d(
            const char* kernel_name,
            size_t n_thrgps_x, size_t n_thrgps_y, size_t n_thrs,
            const void* params, size_t params_size,
            MTL::Buffer** bufs, size_t buf_cnt
        ) {
            auto* enc = g_metal_ctx->cur_encoder;
            auto it = g_metal_ctx->pipelines.find(kernel_name);
            assert(it != g_metal_ctx->pipelines.end());
            auto* pipeline = it->second;

            enc->setComputePipelineState(pipeline);

            size_t start_idx = 0; // some kernels don't have a params struct
            if (params && params_size > 0) {
                enc->setBytes(params, params_size, 0);
                start_idx = 1;
            }
            
            for (size_t i = 0; i < buf_cnt; ++i) {
                auto* buf = static_cast<MTL::Buffer*>(bufs[i]);
                enc->setBuffer(buf, 0, start_idx + i);
            }
            
            MTL::Size grid_size(n_thrgps_x, n_thrgps_y, 1);
            MTL::Size group_size(n_thrs, 1, 1);
            enc->dispatchThreadgroups(grid_size, group_size);
        }

    }
#endif