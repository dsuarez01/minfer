CPU support in progress. Currently, this project supports non-sharded Qwen3 GGUF model files — note the tokenizer data must be present and is assumed to be shipped with the "data gym" style byte mapping used by GPT-2 and other BPE tokenizers, where non-printable bytes are mapped to specific Unicode codepoints. Learn more about this at [Llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

External dependencies:
- [UTF8CPP](https://github.com/nemtrif/utfcpp)
- [PCRE2](https://github.com/PCRE2Project/pcre2)
- [Minja](https://github.com/google/minja)

Current progress so far:

## 9-3-25

Example output with the Qwen3-0.6B-FP32 GGUF model (available [here](https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32)) on Apple's M2 Pro chip:

```
Model: ./gguf/Qwen3-0.6B-FP32.gguf
Input: What is 2+2?
Mode: thinking
Max length: 4096
Seed: 42
Temperature: 0.6
Top-p: 0.95
Top-k: 20
Min-p: 0
Presence penalty: 1.5

Formatted message:
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant

Number of tokens: 18
Prefill progress: 18/18
Prefill time done! Time is: 10.0939 seconds
<think>
Okay, let's see. The user is asking what 2 plus 2 equals. Hmm, I need to figure out the answer here. Well, basic arithmetic problems like this usually have straightforward answers. So adding two 2s together... that should be 4.

Wait, but maybe there's something more to it? Like if they're thinking of some other mathematical concepts or puzzles? But no, in regular math, 2 + 2 is definitely 4. Let me make sure there's no trick here. Maybe a riddle or a different context? No, I don't think so. This seems too simple for a riddle. It's just a simple addition problem. 

So, yeah, the answer should be 4. I don't see any other possibilities here. The user probably wants the basic calculation right away. They might be testing if I can recognize that 2 plus 2 is four without overcomplicating things.
</think>

2 + 2 = 4<|im_end|>

Number of tokens generated: 206 toks
Prefill time: 10.094 sec(s)
Generation throughput: 2.07 tok/sec
Mem. Bandwidth: 6.148 GB/sec
```

Achieved with a naive FP32 matmul implementation, KV cache, and computation buffers. FP16 + BF16 models currently use a matmul that dequantizes scalar elements. Explicit SIMD versions with these dtypes will be implemented soon.

One of the challenges thus far has been with (1) implementing a BPE tokenizer compatible with the GGUF format from scratch, and (2) finding a model with weight sizes in the decoder blocks that fit comfortably within the L3 cache of my laptop.

#### "Data Gym" Byte Mapping Scheme:

(TO-DO: Add me!)

#### Cache Thrashing/Misses:
Running the larger 1.7B/4B models resulted in cache misses that catastrophically slowed down forward passes, as made evident by the CPU profiler available via [Instruments](https://developer.apple.com/tutorials/instruments):

<img width="1297" height="610" alt="Profiler" src="https://github.com/user-attachments/assets/5db10fc2-922a-4afc-8758-4d6294a5632f" />

This time profile was obtained by running `xctrace` on the `generate` method in the `BaseModel` class on Unsloth's `Qwen3-4B-Thinking-2507-F16` GGUF model. The largest MoE tensors take up ~49.8 MB of memory, which resulted in cache misses/thrashing in the forward pass of the MoE layers.

The solution to this was to simply select a model with decoder block tensors (specifically, MoE tensors) that fit comfortably within the L3 cache on my processor. According to [Notebookcheck](https://www.notebookcheck.net/Apple-M2-Pro-Processor-Benchmarks-and-Specs.682450.0.html), the L3 cache of the M2 Pro processor has a 24 MB capacity. I found that `Qwen3-0.6B-FP32` was the model with the largest parameter count that I was able to run at full precision comfortably (~12.6 MB for the largest MoE tensors), and am therefore making use of it for these test runs.

#### Speed-of-light / Ideal Throughput Computation:
- A theoretical estimate of the maximum throughput in tokens/sec we can achieve, given that model inference is memory bound.
- We need the memory bandwidth of the processor and the size of the model + KV cache memory per position (token) to determine this:
  - Memory bandwidth of my processor is [200 GB/sec](https://www.notebookcheck.net/Apple-M2-Pro-Processor-Benchmarks-and-Specs.682450.0.html)
  - Model size is `6e8 params * 4 bytes/param = 2.4e9 bytes/token`. (For each token in the forward pass, we have to read 2.4e9 bytes of memory.)
  - KV cache size per position (token) is `2 * 36 * 8 * 128 * 4 (2 * n_layers * n_kv_heads * d_head * sizeof(float)) = 294912 bytes/token`
  - In total, we read `2400000000 + 294912 = 2400294912 bytes/token`
- Thus, the ideal upper bound on the throughput we can achieve is: `200e9 bytes/sec / 2400294912 bytes/token = 83.3 tokens/sec`.

Due to the unified memory architecture on Apple silicon, where a single memory controller serves both the CPU and GPU, the calculation of the speed-of-light is _identical_ for the GPU. There are limitations that will prevent us from achieving this sort of throughput: throttling and system/scheduling overhead come to mind. However, it still serves as a useful bound for how efficiently we are serving memory to the computational units.

#### Checklist of Improvements (optimizations to be implemented, etc.):
- [ ] Threading in the FP32 matmul implementation
- [ ] Explicit SIMD for the FP16, BF16 matmuls (with -O3 the compiler might already implement this)
- [ ] Head-level and expert-level parallelization
- [ ] Quantizing KV cache
- [ ] Implementing operations for the GPU (put this in the checklist for the feature/gpu branch once relevant)
- [ ] Refactor loader, tokenizer, chat template handling

Disclaimer: use the loader with reputable model providers on HuggingFace, e.g. Unsloth). This is meant to be an educational implementation, although you are welcome to fork and try it out for yourself. Note that I am not responsible for the consequences of any misuse.
