### Minfer: minimal inference engine running on Apple M-series chips

Currently, this project supports non-sharded Qwen3 GGUF model files that come bundled with the corresponding GGUF tokenizer data.

### Precision Support

|     (CPU-Only Inference)     | FP32 Support | FP16 Support | BF16 Support  |
|------------------------------|--------------|--------------|---------------|
| M1 Series                    | Y ✅         | Y ✅         |     N ❌       |
| M2 Series                    | Y ✅         | Y ✅         |     Y ✅       |
| M3 Series                    | Y ✅         | Y ✅         |     Y ✅       |
| M4 Series                    | Y ✅         | Y ✅         |     Y ✅       |


### Usage Instructions

Clone the repo and run the following instructions:

```bash
# Skip if already installed
brew install libomp

# Clone the repository
git clone https://github.com/dsuarez01/minfer.git
cd minfer

# Init. and update git submodules (for PCRE2)
git submodule update --init --recursive

# Build
cmake -S . -B build # NOTE: -DCMAKE_BUILD_TYPE is Release by default, pass in Debug if needed
cmake --build build --parallel

# Tests
./build/tests/test_ops
./build/tests/qwen3/test_tokenizer <path_to_gguf_file>

# App usage instructions
./build/apps/benchmark -h
./build/apps/generate -h

# Examples (for optimal performance, set OMP_NUM_THREADS=<# of p-cores>):
OMP_NUM_THREADS=6 ./build/apps/benchmark model.gguf -s 42
OMP_NUM_THREADS=6 ./build/apps/generate model.gguf -p "Hello" -m 2048 -s 42 -i
```

### Benchmark Performance

The benchmark consists of a 512-token prefill and 128-token generation phase (referenced in the table as `tg-128`, refer to the statistics `llama-bench` reports. More info can be found at the [llama.cpp](https://github.com/ggml-org/llama.cpp) project repository). 

This repo currently doesn't support batch decoding for tokens. Thus, the prefill statistics are not competitive with respect to existing inference engine implementations and are therefore omitted from the results reported here. Tested with 4k max context, seed 30 on 2023 M2 Pro Macbook Pro (6 performance cores, 4 efficiency in the ["binned model"](https://en.wikipedia.org/wiki/Apple_M2)).

|     (M2 Pro CPU, avg of 5 runs)     |      Minfer tg-128 (tok/s)        |    Llama-bench tg-128 (tok/s)    |
|-------------------------------------|-----------------------------------|----------------------------------|
| Qwen3-0.6B-FP32                     |               37.5                |           46.52 ± 0.46           |
| Qwen3-0.6B-BF16                     |               57.2                |              N/A^*               |
| Qwen3-1.7B-BF16                     |               27.0                |              N/A^*               |

* llama.cpp does not support BF16 CPU-only inference for M2 Pro / the associated ISA.

Refer to e.g. [Unsloth AI](https://huggingface.co/unsloth) for access to the recommended GGUF model sizes and precisions listed above. I've yet to find FP16 versions of the 0.6B and 1.7B models, but will test them if and when they become available. Due to cache layer sizing (see the stats for the [M2 Pro](https://en.wikipedia.org/wiki/Apple_M2), for example), I found that these were the only models that could be reliably tested (without cache misses, thrashing, etc.).

#### Checklist of Improvements (optimizations to be implemented, etc.):
- [x] Threading in the naive FP32 matmul implementation
- [x] Head-level parallelization
- [x] Manual SIMD for the FP32, FP16, BF16 matmuls using NEON intrinsics
- [x] Refactor loader, tokenizer, improve chat template handling
- [ ] Quantizing KV cache (will implement for GPU)
- [ ] Implementing operations for the GPU
- [ ] Add support for multi-turn conversation

#### External dependencies:
- [PCRE2](https://github.com/PCRE2Project/pcre2) 
- [Minja](https://github.com/google/minja)

#### Acknowledgements:
- [andrewkchan/yalm](https://github.com/andrewkchan/yalm)
- [zeux/calm](https://github.com/zeux/calm)
- The excellent blogposts associated with both of the above projects

#### Disclaimer:
This project is purely intended for educational purposes only.

#### Additional remarks:
Inference is not necessarily memory-bound with smaller models. Calculating the ideal throughput based on the mem bandwidth (200 GB/s) on my M2 Pro chip shows that max throughput is ~80 tok/s for `Qwen3-0.6B-FP32` with 4k max context. We only achieve ~38-46 tok/s in practice, so it is clear that we are compute bound. 

Most of the speed gain here comes from (1) taking advantage of SIMD on FP32, FP16, BF16 (ARM Neon has intrinsics that support SIMD on all of these datatypes, depending on the processor and version of ARM you compile with: see function `apply_arch_flags` in `./CMakeLists.txt` and adjust the flags as necessary), and (2) utilizing ILP by sending out N independent FMAs per CPU cycle. where N is the number of vector pipelines in the processor that can run simultaneously (in my case, N=4 for the M2 Pro). See more about this at [https://github.com/philipturner/metal-benchmarks]. I am not exactly sure what makes llama.cpp faster, but it appears that they've recently pushed for faster CPU inference, necessitating CPU matmul kernels in hand-written in assembly (no more fighting the compiler!).

Other processors will see faster CPU-only inference performance, due to support for wider SIMD registers (e.g. AVX2 w/ 512-bit). Other alternatives to better utilize processor performance include [AMX](https://zhen8838.github.io/2024/04/23/mac-amx_en) and the Accelerate framework offering built-in GEMV and GEMM operations... But the former is horribly undocumented, while the latter seems to defeat the purpose of an educational implementation. (I have plans to come back to AMX eventually.)

Another tool I found to be useful in profiling CPU activity is [Instruments](https://en.wikipedia.org/wiki/Instruments_(software)), a tool (bundled with XCode 3.0+) that allows you to run compiled binaries as shown below:

<img width="1370" height="792" alt="profiling" src="https://github.com/user-attachments/assets/1141acbe-dff1-41cd-b413-88936af423dd" />

GPU support via MSL is being implemented next. I suspect that we should be able to support larger models, the constraint being that the model must fit in VRAM. At some point, I will consider how to implement support for other ISAs and CUDA (the focus here is to get the implementation running on my own laptop first).
