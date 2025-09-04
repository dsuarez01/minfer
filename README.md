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

Running the larger 1.7B/4B models resulted in cache misses that catastrophically slowed down forward passes, as made evident by the CPU profiler available via [Instruments](https://developer.apple.com/tutorials/instruments):

<img width="1297" height="610" alt="Profiler" src="https://github.com/user-attachments/assets/5db10fc2-922a-4afc-8758-4d6294a5632f" />

This time profile was obtained by running `xctrace` on the `generate` method in the `BaseModel` class on a Qwen3-4b model. The largest MoE tensors take up ~49.8 MiB of memory, which resulted in cache misses/thrashing in the forward pass of the MoE layers.

The solution to this was to simply select a model with decoder block tensors (specifically, MoE tensors) that fit comfortably within the L3 cache on my processor. According to [Notebookcheck](https://www.notebookcheck.net/Apple-M2-Pro-Processor-Benchmarks-and-Specs.682450.0.html), the L3 cache of the M2 Pro processor has a 24 MB capacity. I found that `Qwen3-0.6B-FP32` was the model with the largest parameter count that I was able to run at full precision comfortably (~12.6 MB for the largest MoE tensors), and am therefore making use of it for these test runs.

Speed-of-light computation: **(TO-DO: add me!)**

Checklist of improvements to be made (optimizations to be implemented, etc.): **(TO-DO: add me!)**
- Refactor the loader + tokenizer
- Improve chat template handling

Disclaimer: use the loader with reputable model providers on HuggingFace, e.g. Unsloth). This is meant to be an educational implementation, although you are welcome to fork and try it out for yourself. Note that I am not responsible for the consequences of any misuse.
