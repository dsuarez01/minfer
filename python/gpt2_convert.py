import argparse
import os
import shutil
import tempfile

from gguf import GGUFReader, GGUFWriter, GGUFValueType

# see https://github.com/zeux/calm/blob/main/tools/convert.py
def create_gpt2_byte_decoder():
    """ Reverses the GPT-2 tokenizer byte mapping """
    # codepoints of printable chars are not remapped
    bs = (list(range(ord("!"), ord("~") + 1)) + 
          list(range(ord("¡"), ord("¬") + 1)) + 
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        # non-printable char codepoints 
        # shifted up by 256 in the mapping
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    
    byte_decoder = {}
    for byte_val, unicode_char in zip(bs, cs):
        byte_decoder[unicode_char] = byte_val # reverse lookup to undo
    
    return byte_decoder

def process_token(token: str) -> bytes:
    """ Reverses GPT-2 codepoint mapping scheme for nonprintable chars """
    if not token:
        return b""

    byte_values = []
    for char in token:
        if char in BYTE_DECODER:
            byte_values.append(BYTE_DECODER[char])
        else:
            raise ValueError(f"Char {char} not found in BYTE_DECODER")
    
    return bytes(byte_values)

def copy_metadata(writer: GGUFWriter, reader: GGUFReader, key: str) -> None:
    """ Copy metadata KV pair from reader to writer """
    field = reader.get_field(key)
    if field is None: 
        return
    
    # gets original val and type info
    value = field.contents()
    if not field.types: 
        return
    
    main_type = field.types[0]
    sub_type = field.types[1] if len(field.types) > 1 and main_type == GGUFValueType.ARRAY else None
    
    writer.add_key_value(key, value, main_type, sub_type)

def main():
    parser = argparse.ArgumentParser(
        description="Modify GGUF file's tokenizer.ggml.tokens and .merges, conditional on tokenizer.ggml.model being 'gpt2'"
    )
    parser.add_argument("gguf_path", type=str, help="Path to the .gguf file")
    args = parser.parse_args()

    try:
        reader = GGUFReader(args.gguf_path)

        tokenizer_model_field = reader.get_field("tokenizer.ggml.model")
        if tokenizer_model_field is None:
            print("No tokenizer model found")
            return
            
        tokenizer_model = tokenizer_model_field.contents()
        if tokenizer_model != "gpt2":
            print(f"Tokenizer model is '{tokenizer_model}', not 'gpt2': no changes made")
            return

        # process tokens if present
        tokens_field = reader.get_field("tokenizer.ggml.tokens")
        if tokens_field is None:
            print("'tokenizer.ggml.tokens' not found in metadata, no changes made")
            return
        
        tokens = tokens_field.contents()
        print(f"Processing {len(tokens)} tokens...")
        new_tokens = [process_token(t) for t in tokens]

        # process merges if present
        merges_field = reader.get_field("tokenizer.ggml.merges")
        if merges_field is None:
            print("'tokenizer.ggml.merges' not found in metadata, no changes made")
            return
        
        merges = merges_field.contents()
        print(f"Processing {len(merges)} merges...")
        new_merges = []
        for m in merges:
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError(f"Invalid merge format: {m}")
            p1 = process_token(parts[0])
            p2 = process_token(parts[1])
            new_merges.append(p1 + b" " + p2)

        # temp file in same directory as GGUF file
        dir_name = os.path.dirname(args.gguf_path)
        with tempfile.NamedTemporaryFile(dir=dir_name, delete=False) as temp_file:
            temp_path = temp_file.name
        
        arch_field = reader.get_field("general.architecture")
        arch = arch_field.contents() if arch_field else "none"
        writer = GGUFWriter(temp_path, arch)

        # writer automatically adds these keys, should be skipped
        auto_generated_keys = {
            "general.architecture",
            "GGUF.version",
            "GGUF.tensor_count",
            "GGUF.kv_count"
        }

        # copy all (potentially modified) metadata, skipping auto-generated keys
        for key, _ in reader.fields.items():
            if key in auto_generated_keys:
                continue
            elif key == "tokenizer.ggml.tokens":
                # Use add_token_list which handles bytes properly
                writer.add_token_list(new_tokens)
            elif key == "tokenizer.ggml.merges":
                writer.add_token_merges(new_merges)
            else:
                copy_metadata(writer, reader, key)
        
        # copy tensors unchanged
        for tensor in reader.tensors:
            writer.add_tensor(
                name=tensor.name,
                tensor=tensor.data,
                raw_shape=tensor.shape,
                raw_dtype=tensor.tensor_type,
            )
        
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # replace orig file with temp file
        shutil.move(temp_path, args.gguf_path)
        print(f"Successfully modified {args.gguf_path} in-place.")
        
    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        print(f"Error: {e}")
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

if __name__ == "__main__":
    BYTE_DECODER = create_gpt2_byte_decoder()
    main()