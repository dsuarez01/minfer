Currently, this project supports non-sharded Qwen3 GGUF model files that come bundled with tokenizer data.

### Precision Support

|     (CPU-Only Inference)     | FP32 Support | FP16 Support | BF16 Support |
|------------------------------|--------------|--------------|--------------|
| M1 Series                    | Y            | Y            |     N        |
| M2 Series                    | Y            | Y            |     Y        |
| M3 Series                    | Y            | Y            |     Y        |
| M4 Series                    | Y            | Y            |     Y        |


### Benchmark Performance

The benchmarks consist of a prefill and token generation phase. The repo currently doesn't support batch decoding for tokens, so the prefill statistics are not competitive with respect to existing implementations and are therefore omitted from the results reported here. 

|     (CPU Only)     |              |
|--------------------|--------------|
| Qwen3-0.6B-FP32    |              |
| Qwen3-0.6B-BF16    |              |
| Qwen3-1.7B-BF16    |              |

Refer to e.g. [Unsloth AI](https://huggingface.co/unsloth) for access to the recommended GGUF model sizes and precisions listed above. I've yet to find FP16 versions of the 0.6B and 1.7B models, but will test them if and when they become available. Due to cache layer sizing (see the stats for the [M2 Pro](https://en.wikipedia.org/wiki/Apple_M2), for example), I found that these were the only models that could be reliably tested (without cache misses, thrashing, etc.).



#### Checklist of Improvements (optimizations to be implemented, etc.):
- [x] Threading in the naive FP32 matmul implementation
- [x] Head-level parallelization
- [x] Manual FP32 SIMD implementation
- [x] Explicit SIMD for the FP32, FP16, BF16 matmuls using NEON intrinsics
- [ ] Quantizing KV cache (will implement for GPU)
- [ ] Implementing operations for the GPU (put this in the checklist for the feature/gpu branch once relevant)
- [ ] Refactor loader, tokenizer, improve chat template handling


Generation and benchmark binaries currently run on the CPU.

External dependencies:
- [PCRE2](https://github.com/PCRE2Project/pcre2)
- [Minja](https://github.com/google/minja)

