# Minfer: Minimal inference engine running on Apple M-series chips


## Table of Contents

* [**Overview**](#overview)
    - [Recommended System Config](#overview---recommended-system-config)
    - [Precision Support](#overview---precision-support)
    - [Usage Instructions](#overview---usage-instructions)

* [**Benchmark Performance**](#benchmark-performance)
    - [CPU](#performance---cpu-inference-results)
    - [GPU](#performance---gpu-inference-results)

* [**Checklist of Improvements**](#checklist-of-improvements)

* [**External Dependencies**](#external-dependencies)

* [**Acknowledgments**](#acknowledgments)

* [**Disclaimer**](#disclaimer)

* [**Additional Remarks**](#additional-remarks)
    - [CPU](#remarks---cpu-inference)
    - [GPU](#remarks---gpu-inference)


## Overview

### Overview - Recommended System Config.

This project currently supports non-sharded Qwen3 GGUF model files that come bundled with the corresponding GGUF tokenizer data.

Recommended OS version, chip set, C++ version, compiler: `>=` MacOS v15 (Sequoia), `>=` M2 chip series, C++17. Project will likely not compile with `g/g++`, so use `clang/clang++` if possible. Note that `clang/clang++` are shipped with the XCode Command Line Tools (CLT) by default, which you can check at `/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin` or similar, depending on your OS version. Ensure that you have the XCode CLT corresponding to your OS version installed. See [Precision Support](#precision-support) section below for more details.

[Work-in-progress: broader support / more robust error handling in the build config, please excuse any errors and/or bugs]

### Overview - Precision Support

|     CPU + GPU Inference      | FP32 Support | FP16 Support |        BF16 Support        |
|------------------------------|--------------|--------------|----------------------------|
| M1 Series                    | Y            | Y            | N (CPU), Y (GPU, emulated) |
| M2 Series                    | Y            | Y            | Y (CPU), Y (GPU, emulated) |
| M3 Series                    | Y            | Y            | Y (CPU), Y (GPU, native)   |
| M4 Series                    | Y            | Y            | Y (CPU), Y (GPU, native)   |

For **CPU support**, I would recommend adjusting the `-march` and `-mtune` flags in `function(apply_arch_flags target_name)` (defined at `./CMakeLists.txt`) to match an ARM Neon version, desired dtype support, and target architecture that is compatible with your system. You'll want to set the flags as follows: `-march=<armv8.X-a>+<fp16 for FP16 support, only supported for armv8.X-a where X>=2>+<bf16 for BF16 support, only supported for armv8.X-a where X>=6>`; `-mtune=<apple-mX>` according to whatever your chip version is (`X=1, X=2, X=3, X=4, etc. as new chips are added`).

For **GPU support**, I would recommend keeping Metal 3.2 as the version (see [this link](https://support.apple.com/en-us/102894) for more information about Metal support across M-series chips + OS versions), and adjusting the `-target` flag in `xcrun -sdk macosx metal -std=metal3.2 -target air64-apple-macos26.0 -c ${METAL_SHADER_SOURCE} -o ${METAL_AIR}` as needed (see `./src/CMakeLists.txt`). The relevant section of commands here will precompile the shader source file (`./src/ops/kernels.metal`) into a .metallib library, which is then embedded as a byte array into the program for use by the Metal interface (credit: [zeux/calm](https://github.com/zeux/calm)).

### Overview - Usage Instructions

Clone the repo and run the following instructions, adjusting the build config if necessary.

(NOTE: for python conversion, refer to [`./python`](./python/README.md#gguf-tokenizer-data-conversion-tool-gpt2_convertpy) for more details on usage.)

```bash
# Skip if already installed, only necessary for threading support on the CPU
brew install libomp

# Clone the repository
git clone https://github.com/dsuarez01/minfer.git
cd minfer

# Init. and update git submodules (for PCRE2)
git submodule update --init --recursive

# Build (see "Precision Support" section for adjustments to arch flags as needed)
cmake -S . -B build # NOTE: -DCMAKE_BUILD_TYPE is Release by default, pass in Debug if needed
cmake --build build --parallel

############ PYTHON CONVERSION SCRIPT ############
# NOTE: Convert the model using the gpt2_convert.py script BEFORE running inference, see ./python for more info
# if uv not installed already
pip install uv

# uv sync in python dir to install env and dependencies
cd python
uv sync

# run from project root
cd ..
uv run --project python python/gpt2_convert.py <path_to_gguf_file>

# optionally, run python/summary.py for a print-out of the metadata
uv run --project python python/summary.py <path_to_gguf_file>
########################

# Tests
./build/tests/cpu_ops/test_cpu_ops
./build/tests/qwen3/test_tokenizer <path_to_gguf_file>

# App usage instructions
./build/apps/benchmark -h
./build/apps/generate -h

# Usage examples for CPU inference (for optimal performance, set OMP_NUM_THREADS=<# of p-cores>):
OMP_NUM_THREADS=6 ./build/apps/benchmark <path_to_gguf_file> -d cpu -s 42
OMP_NUM_THREADS=6 ./build/apps/generate <path_to_gguf_file> -d cpu -p "Hello" -m 4096 -s 42 -i

# Usage examples for GPU inference
./build/apps/benchmark <path_to_gguf_file> -d gpu -s 42
./build/apps/generate <path_to_gguf_file> -d gpu -p "Hello" -m 4096 -s 42 -i
```


## Benchmark Performance

The benchmark consists of a 512-token prefill and 128-token generation phase (referenced in the table as `tg-128`, refer to the statistics `llama-bench` reports. More info can be found at the [llama.cpp](https://github.com/ggml-org/llama.cpp) project repository). 

### Performance - CPU Inference Results

This repo currently doesn't support batch decoding for tokens. Thus, the prefill statistics are not competitive with respect to existing inference engine implementations, and are therefore omitted from the results reported here. Tested with 4k max context, seed 30 on 2023 M2 Pro Macbook Pro (6 performance cores, 4 efficiency in the ["binned model"](https://en.wikipedia.org/wiki/Apple_M2)).

|     (M2 Pro CPU, avg of 5 runs)     |      Minfer tg-128 (tok/s)        |    Llama-bench tg-128 (tok/s)    |
|-------------------------------------|-----------------------------------|----------------------------------|
| Qwen3-0.6B-FP32                     |               37.5                |           46.52 ± 0.46           |
| Qwen3-0.6B-BF16                     |               57.2                |              N/A*                |
| Qwen3-1.7B-BF16                     |               27.0                |              N/A*                |

* llama.cpp does not support BF16 CPU-only inference for M2 Pro / the associated ISA.

Refer to [Unsloth AI](https://huggingface.co/unsloth) or other reputable model providers for access to the recommended GGUF model sizes and precisions listed above, and ensure that you've converted the models before testing. I've yet to find FP16 versions of the 0.6B and 1.7B models, but will test them if and when they become available. Due to cache layer sizing (see the stats for the [M2 Pro](https://en.wikipedia.org/wiki/Apple_M2), for example), I found that these were the only models that could be reliably tested (without cache misses, thrashing, etc.).

### Performance - GPU Inference Results

|     (M2 Pro GPU, avg of 5 runs)     |      Minfer tg-128 (tok/s)        |    Llama-bench tg-128 (tok/s)    |
|-------------------------------------|-----------------------------------|----------------------------------|
| Qwen3-0.6B-BF16                     |                                   |                                  |
| Qwen3-0.6B-FP16                     |                                   |                                  |
| Qwen3-0.6B-FP32                     |                                   |                                  |
| Qwen3-1.7B-BF16                     |                                   |                                  |
| Qwen3-4B-Instruct-2507-FP16         |                                   |                                  |

Refer to [Unsloth AI](https://huggingface.co/unsloth) or other reputable model providers for access to the recommended GGUF model sizes and precisions listed above, and ensure that you've converted the models (refer to the [Python script README](./python/README.md#gguf-tokenizer-data-conversion-tool-gpt2_convertpy)) before testing. I would test larger models, but am limited by the RAM available on my GPU: the [recommended max working set size](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html) — which functions more as a hard limit on mem. usage, as noted in the article — is 75% of the available physical RAM, so ~12 GB for 16 GB system, ~24GB for a 32 GB system, etc.


## Checklist of Improvements:
- [x] Threading in the naive FP32 matmul implementation
- [x] Head-level parallelization
- [x] Manual SIMD for the FP32, FP16, BF16 matmuls using NEON intrinsics
- [x] Refactor loader, tokenizer, improve chat template handling
- [x] Implementing operations for the GPU
- [ ] Quantizing KV cache
- [ ] Add support for multi-turn conversation
- [ ] Refactor code to support more models, tokenizers, activation types, compilers, ISAs, OS versions, etc.
- [ ] CPU offloading if model exceeds max working set size, or assertion (refer back to metal-cpp)
- [ ] (Eventually) Support more GGML quantization types (e.g. Q8_0)


## External Dependencies:
- [PCRE2](https://github.com/PCRE2Project/pcre2) 
- [Minja](https://github.com/google/minja)
- [Metal-cpp] (https://github.com/bkaradzic/metal-cpp)


## Acknowledgments:
- [andrewkchan/yalm](https://github.com/andrewkchan/yalm)
- [zeux/calm](https://github.com/zeux/calm)
- The excellent blogposts associated with both of the above projects


## Disclaimer:
This project is purely intended for educational purposes only. I do not assume any responsibility for misuse.


## Additional Remarks:

### Remarks - CPU Inference:
Inference is not necessarily memory-bound with smaller models. Calculating the ideal throughput based on the mem bandwidth (200 GB/s) on my M2 Pro chip shows that max throughput is ~80 tok/s for `Qwen3-0.6B-FP32` with 4k max context. We only achieve ~38-46 tok/s in practice on the CPU, so it is clear that we are compute bound. 

Most of the speed gain here comes from (1) taking advantage of SIMD on FP32, FP16, BF16 (ARM Neon has intrinsics that support SIMD on all of these datatypes, depending on the processor and version of ARM you compile with: see function `apply_arch_flags` in `./CMakeLists.txt` and adjust the flags as necessary), and (2) utilizing ILP by sending out N independent FMAs per CPU cycle. where N is the number of vector pipelines in the processor that can run simultaneously (in my case, N=4 for the M2 Pro). See more about this at [https://github.com/philipturner/metal-benchmarks]. I am not entirely sure what makes llama.cpp faster as compared to my current implementation, but it appears that they've recently pushed for faster CPU inference, necessitating CPU matmul kernels in hand-written in assembly (no need to fight the compiler!). It should be easier to match their performance using the GPU.

Other processors will see faster CPU-only inference performance, due to ISA support for wider SIMD registers (e.g. AVX2 w/ 512-bit). Other alternatives to better utilize processor performance include [AMX](https://zhen8838.github.io/2024/04/23/mac-amx_en) and the Accelerate framework offering built-in GEMV and GEMM operations... But the former is horribly undocumented, while the latter seems to defeat the purpose of an educational implementation. (It would be worth revisiting AMX, and I've considered doing so.)

Another tool I found to be useful in profiling CPU activity is [Instruments](https://en.wikipedia.org/wiki/Instruments_(software)), a tool (bundled with XCode 3.0+) that allows you to run compiled binaries as shown below:

<img width="1370" height="792" alt="profiling" src="https://github.com/user-attachments/assets/1141acbe-dff1-41cd-b413-88936af423dd" />

### Remarks - GPU Inference:
\[ TO-DO: add remarks about GPU support here \]

At some point, I will consider how to implement support for other ISAs, CUDA, compilers, etc. But the ultimate goal here is to first get the implementation running on my own laptop.