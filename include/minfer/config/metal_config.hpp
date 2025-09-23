#pragma once

#include <cstddef>

// interface through which methods interact with the device
namespace MetalManager {
    void init();
    void* upload(void* data, size_t size);
    void release(void* metal_buffer);
    // TO-DO: cmd buffer management, kernel compilation, etc.
}