Currently, this project supports non-sharded Qwen3 GGUF model files that come bundled with the corresponding GGUF tokenizer data.

### Precision Support

|     (CPU-Only Inference)     | FP32 Support | FP16 Support | BF16 Support  |
|------------------------------|--------------|--------------|---------------|
| M1 Series                    | Y ✅         | Y ✅         |     N ❌     |
| M2 Series                    | Y ✅         | Y ✅         |     Y ✅     |
| M3 Series                    | Y ✅         | Y ✅         |     Y ✅     |
| M4 Series                    | Y ✅         | Y ✅         |     Y ✅     |


### Usage Instructions

Clone the repo and run the following instructions:

```bash
# Clone the repository
git clone https://github.com/dsuarez01/minfer.git
cd minfer

# Init., update git submodules (for PCRE2)
git submodule update --init --recursive

# Build
cmake -S . -B build # NOTE: -DCMAKE_BUILD_TYPE is Release by default, pass in Debug if needed
cmake --build build --parallel

(include more usage instructions below)
```


### Benchmark Performance

The benchmark consists of a 512-token prefill and 128-token generation phase (referenced in the table as `tg-128`, refer to the statistics `llama-bench` reports. More info can be found at the [llama.cpp](https://github.com/ggml-org/llama.cpp) project repository). 

This repo currently doesn't support batch decoding for tokens. Thus, the prefill statistics are not competitive with respect to existing inference engine implementations and are therefore omitted from the results reported here. 

|     (CPU Only)     |      Minfer tg-128        |    Llama-bench tg-128     |
|--------------------|---------------------------|---------------------------|
| Qwen3-0.6B-FP32    |                           |                           |
| Qwen3-0.6B-BF16    |                           |                           |
| Qwen3-1.7B-BF16    |                           |                           |

Refer to e.g. [Unsloth AI](https://huggingface.co/unsloth) for access to the recommended GGUF model sizes and precisions listed above. I've yet to find FP16 versions of the 0.6B and 1.7B models, but will test them if and when they become available. Due to cache layer sizing (see the stats for the [M2 Pro](https://en.wikipedia.org/wiki/Apple_M2), for example), I found that these were the only models that could be reliably tested (without cache misses, thrashing, etc.).

#### Checklist of Improvements (optimizations to be implemented, etc.):
- [x] Threading in the naive FP32 matmul implementation
- [x] Head-level parallelization
- [x] Manual SIMD for the FP32, FP16, BF16 matmuls using NEON intrinsics
- [ ] Quantizing KV cache (will implement for GPU)
- [ ] Implementing operations for the GPU (put this in the checklist for the feature/gpu branch once relevant)
- [ ] Refactor loader, tokenizer, improve chat template handling

External dependencies:
- [PCRE2](https://github.com/PCRE2Project/pcre2)
- [Minja](https://github.com/google/minja)
