#pragma once

#include <cstddef>
#include <cassert>

// interface through which methods interact with the device
namespace MetalManager {
    #ifdef USE_METAL

        void init();

        // cmd buffer management
        void begin_frame();           // starts new cmd buffer + encoder
        void end_frame();             // commit, wait
        
        void dispatch1d(
            const char* kernel_name,
            size_t n_thrgps, size_t n_thrs,
            const void* params, size_t params_size,
            void** bufs, size_t buf_cnt
        );

        void dispatch2d(
            const char* kernel_name,
            size_t n_thrgps_x, size_t n_thrgps_y, size_t n_thrs,
            const void* params, size_t params_size,
            void** bufs, size_t buf_cnt
        );

        void* buf_contents(void* buffer_ptr);

    #endif
    // stubs
    #ifndef USE_METAL
        void init() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void begin_frame() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void end_frame() { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void dispatch1d(const char*, size_t, size_t, const void*, size_t, void**, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void dispatch2d(const char*, size_t, size_t, size_t, const void*, size_t, void**, size_t) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
        void* buf_contents(void*) { assert(false && "Metal backend not detected, this stub should never be called"); return; }
    #endif
}