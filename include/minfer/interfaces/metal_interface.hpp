#pragma once

#include <cstddef>
#include <cassert>

namespace MTL { class Buffer; }

// interface through which methods interact with the device
namespace MetalManager {
    #ifdef USE_METAL

        // resource management
        MTL::Buffer* upload(void* cpu_data, size_t size);
        void release(MTL::Buffer* buf);
        void* cpu_ptr(MTL::Buffer* buf);

        // dispatch-related
        void init();

        // cmd buffer management
        void begin_frame(); // starts new cmd buffer + encoder          
        void end_frame(); // commit, wait
        
        void dispatch1d(
            const char* kernel_name,
            size_t n_thrgps, size_t n_thrs,
            const void* params, size_t params_size,
            MTL::Buffer**, size_t buf_cnt
        );

        void dispatch2d(
            const char* kernel_name,
            size_t n_thrgps_x, size_t n_thrgps_y, size_t n_thrs,
            const void* params, size_t params_size,
            MTL::Buffer**, size_t buf_cnt
        );

    #endif
    // stubs
    #ifndef USE_METAL
        // resource management
        inline MTL::Buffer* upload(void*, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return nullptr; }
        inline void release(MTL::Buffer*) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        inline void* cpu_ptr(MTL::Buffer*) { assert (false && "Metal backend not detected, this stub should never be called"); return nullptr; }

        // dispatch-related
        inline void init() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        inline void begin_frame() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        inline void end_frame() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        inline void dispatch1d(const char*, size_t, size_t, const void*, size_t, MTL::Buffer**, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        inline void dispatch2d(const char*, size_t, size_t, size_t, const void*, size_t, MTL::Buffer**, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
    #endif
}