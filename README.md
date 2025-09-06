**CPU support in progress.**

Currently, this project supports non-sharded Qwen3 GGUF model files — note the tokenizer data must be present and is assumed to be shipped with the "data gym" style byte mapping used by GPT-2 and other BPE tokenizers, where non-printable bytes are mapped to specific Unicode codepoints. Learn more about this by referencing the GGUF tokenizer implementation at [Llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

External dependencies:
- [UTF8CPP](https://github.com/nemtrif/utfcpp)
- [PCRE2](https://github.com/PCRE2Project/pcre2)
- [Minja](https://github.com/google/minja)

Current progress so far:

## 9-3-25

Example output with the Qwen3-0.6B-FP32 GGUF model (available [here](https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32)) on Apple's M2 Pro chip:

```
d@Mac minfer % OMP_NUM_THREADS=1 ./build/apps/generate ./gguf/Qwen3-0.6B-FP32.gguf -i "When we mix red and blue, what color do we get?" -m 4096 -s 30 -t 
Model: ./gguf/Qwen3-0.6B-FP32.gguf
Input: When we mix red and blue, what color do we get?
Mode: thinking
Max length: 4096
Seed: 30
Temperature: 0.6
Top-p: 0.95
Top-k: 20
Min-p: 0
Presence penalty: 1.5

Formatted message:
<|im_start|>user
When we mix red and blue, what color do we get?<|im_end|>
<|im_start|>assistant

Number of tokens: 24
Prefill progress: 24/24
Prefill time done! Time is: 11.6998 seconds
<think>
Okay, the user is asking what color we get when mixing red and blue. Let me think about this.

First, I remember that mixing colors can lead to different hues depending on how they're combined. Red and blue are both primary colors in the traditional subtractive pigment model (like paint). When you mix them, the result should be a secondary color, right? But wait, sometimes people might mix them in a way that's not additive, like using light.

Wait, but actually, the answer depends on the method. If you mix two pigments together, the resulting color is called a secondary color. For example, red and blue make violet. But if you add light, it could change. However, the question doesn't specify whether it's additive or subtractive. But since the question is straightforward, maybe they just want the standard answer.

In most cases, mixing red and blue gives violet. But I should check if there's any other possibility. Maybe green? No, because green is derived from mixing yellow and blue. So the correct answer is violet.
</think>

When mixing red and blue, the resulting color is **violet**. This is based on the traditional subtractive color model where mixing primary colors results in secondary colors.<|im_end|>

Number of tokens generated: 255 toks
Prefill time: 11.700 sec(s)
Generation throughput: 2.08 tok/sec
Mem. Bandwidth: 6.285 GB/sec
```

Achieved with a naive FP32 matmul implementation, KV cache, and computation buffers. FP16 + BF16 models currently use a matmul that dequantizes scalar elements. Explicit SIMD versions with these dtypes will be implemented soon.

Challenges thus far have been with (1) implementing a BPE tokenizer compatible with the GGUF format from scratch, and (2) finding a model with weight sizes in the decoder blocks that fit comfortably within the L3 cache of my laptop.

#### "Data Gym" Byte Mapping Scheme:

(TO-DO: Add me!)

#### Cache Thrashing/Misses:
Running the larger 1.7B/4B models resulted in cache misses that catastrophically slowed down forward passes, as made evident by the CPU profiler available via [Instruments](https://developer.apple.com/tutorials/instruments):

<img width="1297" height="610" alt="Profiler" src="https://github.com/user-attachments/assets/5db10fc2-922a-4afc-8758-4d6294a5632f" />

This time profile was obtained by running `xctrace` on the `generate` executable (testing the `generate` method in the `BaseModel` class, i.e. a model text generation task) on Unsloth's `Qwen3-4B-Thinking-2507-F16` GGUF model. The largest MoE tensors take up ~49.8 MB of memory, exceeding the L3 cache size on my processor. This resulted in cache misses/thrashing in the forward pass at the MoE layers.

The solution to this was to learn more about the various cache levels on my processor, and select a model with weights (specifically, MoE tensors) of a size at most the L3 cache capacity.

Via the same profiling tool as mentioned earlier, I noticed that the OS distributes work out to the cores by scheduling thread(s) across 6 specific cores:

<img width="1035" height="605" alt="scheduling" src="https://github.com/user-attachments/assets/6cc6d031-99e2-41bb-ba52-ff3acedb016b" />

According to [Wikipedia](https://en.wikipedia.org/wiki/Apple_M2), my device has 6 performance and 4 efficiency cores. The names are essentially what they imply: efficiency cores are set aside for lighter workloads / background tasks, and free up the performance cores to focus on more compute-intensive tasks such as ours.

Each performance core on the M2 Pro has a 16 MB L2 cache capacity and a 24 MB system-level cache (effectively an L3/last-level cache) shared by all cores, as well as the GPU. 

With this information in mind, I found that `Qwen3-0.6B-FP32`, with ~12.6 MB for the largest MoE tensors, was the model with the largest parameter count that I was able to run at full precision without noticeable delay in the forward pass of the decoder blocks. I am therefore making use of this model for these test runs.

#### Speed-of-light / Ideal Throughput Computation:
- A theoretical estimate of the maximum throughput in tokens/sec we can achieve, given that model inference is memory bound.
- We need the memory bandwidth of the processor and the size of the model + KV cache memory per position (token) to determine this:
  - Memory bandwidth of my processor is [200 GB/sec](https://en.wikipedia.org/wiki/Apple_M2)
  - Model size is `6e8 params * 4 bytes/param = 2.4e9 bytes/token`. (For each token in the forward pass, we have to read 2.4e9 bytes of memory.)
  - KV cache size per position (token) is `2 * 36 * 8 * 128 * 4 (2 * n_layers * n_kv_heads * d_head * sizeof(float)) = 294912 bytes/token`
  - In total, we read `2400000000 + 294912 = 2400294912 bytes/token`
- Thus, the ideal upper bound on the throughput we can achieve is: `200e9 bytes/sec / 2400294912 bytes/token = 83.3 tokens/sec`.

Due to the unified memory architecture on Apple silicon, where the CPU and GPU share memory controllers, the calculation of the ideal throughput is identical for the GPU. There are obviously limitations that will prevent us from achieving the ideal, but it still serves as a useful reference point to determine (1) how well we address bottlenecks in the code to (2) effectively utilize the memory bandwidth.

#### Checklist of Improvements (optimizations to be implemented, etc.):
- [ x ] Threading in the naive FP32 matmul implementation
- [ ] Head-level and expert-level parallelization
- [ ] Explicit SIMD for the FP16, BF16 matmuls
- [ ] Quantizing KV cache
- [ ] Implementing operations for the GPU (put this in the checklist for the feature/gpu branch once relevant)
- [ ] Refactor loader, tokenizer, improve chat template handling

**Disclaimer:** use the loader with reputable model providers on HuggingFace, e.g. Unsloth). This is meant to be an educational implementation, although you are welcome to fork and try it out for yourself. Note that I am not responsible for the consequences of any misuse.

### 9-6-25

Added multithreading to the naive FP32 matrix multiplication. 

Note that all benchmarks have been run with the release build, and -O3 compiler optimization flag. In my implementation, this means that the naive FP32 matmul has been auto-vectorized with SIMD instructions, which allows for better core utilization. Multithreading allows for the work to be distributed across more cores via scheduling, and the auto-vectorization in this case allows for better utilization of each core threads are scheduled to run on.

We might be able to beat the auto-vectorization the compiler tries, though I haven't made an attempt at it yet.

Results for 16 threads (tune this as needed) shown below:

```
d@Mac minfer % OMP_NUM_THREADS=16 ./build/apps/generate ./gguf/Qwen3-0.6B-FP32.gguf -i "When we mix red and blue, what color do we get?" -m 4096 -s 30 -t
Model: ./gguf/Qwen3-0.6B-FP32.gguf
Input: When we mix red and blue, what color do we get?
Mode: thinking
Max length: 4096
Seed: 30
Temperature: 0.6
Top-p: 0.95
Top-k: 20
Min-p: 0
Presence penalty: 1.5

Formatted message:
<|im_start|>user
When we mix red and blue, what color do we get?<|im_end|>
<|im_start|>assistant

Number of tokens: 24
Prefill progress: 24/24
Prefill time done! Time is: 3.20817 seconds
<think>
Okay, the user is asking what color we get when mixing red and blue. Let me think about this.

First, I remember that mixing colors can lead to different hues depending on how they're combined. Red and blue are both primary colors in the traditional subtractive pigment model (like paint). When you mix them, the result should be a secondary color, right? But wait, sometimes people might mix them in a way that's not additive, like using light.

Wait, but actually, the answer depends on the method. If you mix two pigments together, the resulting color is called a secondary color. For example, red and blue make violet. But if you add light, it could change. However, the question doesn't specify whether it's additive or subtractive. But since the question is straightforward, maybe they just want the standard answer.

In most cases, mixing red and blue gives violet. But I should check if there's any other possibility. Maybe green? No, because green is derived from mixing yellow and blue. So the correct answer is violet.
</think>

When mixing red and blue, the resulting color is **violet**. This is based on the traditional subtractive color model where mixing primary colors results in secondary colors.<|im_end|>

Number of tokens generated: 255 toks
Prefill time: 3.208 sec(s)
Generation throughput: 10.06 tok/sec
Mem. Bandwidth: 29.588 GB/sec
```
