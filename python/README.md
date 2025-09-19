## GGUF Tokenizer Data Conversion Tool (gpt2_convert.py)

A convenience utility that reverses the (Unicode) codepoint mapping "hack" GPT-2 tokenizers commonly employ, where non-printable chars have codepoints that are shifted up by 256 (to printable codepoints for easier debugging/viewing, presumably). The tool undoes this mapping for the values located at the KV metadata attributes `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` in the original GGUF file; note that it effectively modifies the GGUF file **in-place**. Ensure that the attribute `tokenizer.ggml.model` exists and is equal to `gpt2`.

Steps to use from the project root:

```bash
pip install uv (if not installed already)
cd python
uv sync
cd ..
uv run python/gpt2_convert.py <path_to_gguf_file>
```

This will result in an identically-named GGUF file with the modified tokenizer data in the same parent directory as `<path_to_gguf_file>`. As a quick check, run the tokenizer test on the modified GGUF file via `./build/tests/test_tokenizer <path_to_gguf_file>`: all of the tests should pass.
