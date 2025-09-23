## GGUF Tokenizer Data Conversion Tool (gpt2_convert.py)

A convenience utility that reverses the (Unicode) codepoint mapping "hack" GPT-2 tokenizers commonly employ, where non-printable chars have codepoints that are shifted up by 256 (to printable codepoints for easier debugging/viewing, presumably). The tool undoes this mapping for the values located at the KV metadata attributes `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` in the original GGUF file; note that it effectively modifies the GGUF file **in-place**. Ensure that the attribute `tokenizer.ggml.model` exists and is equal to `gpt2`.

Steps to use from the project root:

```bash
# if uv not installed already
pip install uv

# sync uv to install env and dependencies
cd python
uv sync

# run from project root
cd ..
uv run python/gpt2_convert.py <path_to_gguf_file>
```

The new GGUF file with the modified tokenizer data will replace the GGUF file located at `<path_to_gguf_file>`. As a quick check for correctness, run the tokenizer test on the modified GGUF file via `./build/tests/<model_name>/test_tokenizer <path_to_gguf_file>`: all of the tests should pass.
