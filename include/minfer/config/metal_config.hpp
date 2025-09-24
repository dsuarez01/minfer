#pragma once

#include <cstddef>
#include <cassert>

// interface through which methods interact with the device
namespace MetalManager {
    #ifdef USE_METAL

        void init();
        void* upload(void* data, size_t size);
        void release(void* metal_buffer);

        // cmd buffer management
        void begin_frame();           // starts new cmd buffer + encoder
        void* get_compute_encoder();  // (returns MTL::ComputeCommandEncoder*)
        void end_frame();             // commit, wait

        // compute pipeline
        void* create_pipeline(const char* kernel_src, const char* fcn_name);
        void dispatch_kernel(void* pipeline, void* encoder, size_t threads_x, size_t threads_y = 1, size_t threads_z = 1);

    #endif
    // stubs
    #ifndef USE_METAL
        void init() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void* upload(void*, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return nullptr; }
        void release(void*) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void begin_frame() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void* get_compute_encoder() { assert(false && "Metal backend not detected, this stub should never be called"); return nullptr; }
        void end_frame() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void* create_pipeline(const char*, const char*) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void dispatch_kernel(void*, void*, size_t, size_t, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
    #endif
}