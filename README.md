CPU support in progress. Currently, this project supports non-sharded Qwen3 GGUF model files.

External dependencies:
- UTF8CPP [https://github.com/nemtrif/utfcpp]
- PCRE2 [https://github.com/PCRE2Project/pcre2]
- Minja [https://github.com/google/minja]

Current progress so far:

- Example output with the Qwen3-0.6B-FP32 GGUF model (available at https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32) on M2 Pro chip **(TO-DO: add usage information for generate)**:

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
Mem. Bandwidth: 5.727 GiB/sec
```

Achieved with a naive FP32 matmul implementation, KV cache, and computation buffers. FP16 + BF16 models currently use a matmul that dequantizes scalar elements. Explicit SIMD versions with these dtypes will be implemented soon.

Speed-of-light computation: **(TO-DO: add me!)**

Checklist of improvements to be made (optimizations to be implemented, etc.): **(TO-DO: add me!)**
- Refactor the loader + tokenizer
- Improve chat template handling

Disclaimer: use the loader with reputable model providers on HuggingFace, e.g. Unsloth). This is meant to be an educational implementation, although you are welcome to fork and try it out for yourself. Note that I am not responsible for the consequences of any misuse.
