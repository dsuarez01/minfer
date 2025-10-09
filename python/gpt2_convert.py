import argparse
import os
import shutil
import tempfile

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType, ReaderField

# see https://github.com/zeux/calm/blob/main/tools/convert.py
def create_gpt2_byte_decoder():
    """ Reverses the GPT-2 tokenizer byte mapping """
    # codepoints of printable chars not remapped
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
    byte_encoder = {}
    for byte_val, unicode_char in zip(bs, cs):
        byte_decoder[unicode_char] = byte_val # reverse lookup: undo
        byte_encoder[byte_val] = unicode_char # forward lookup: apply
    
    return byte_decoder, byte_encoder

def decode_token(token: str) -> bytes:
    """ Converts GPT-2 encoded string to raw bytes """
    if not token:
        return b""

    byte_values = []
    for char in token:
        if char in BYTE_DECODER:
            byte_values.append(BYTE_DECODER[char])
        else:
            raise ValueError(f"Char {char} not found in BYTE_DECODER")
    
    return bytes(byte_values)

def encode_token(token_bytes: bytes) -> str:
    """ Converts raw bytes to GPT-2 encoded string """
    if not token_bytes:
        return ""
    
    chars = []
    for byte_val in token_bytes:
        if byte_val in BYTE_ENCODER:
            chars.append(BYTE_ENCODER[byte_val])
        else:
            raise ValueError(f"Byte {byte_val} not found in BYTE_ENCODER")
    
    return ''.join(chars)

# necessary since reader always tries to utf-8 decode the tokens + merges fields
def get_tokens_or_merges(field : ReaderField) -> list[str] | list[bytes]:
    """ Safely extract tokens/merges field, handles both string and bytes """
    if field is None:
        return None
    
    if field.types and field.types[0] == GGUFValueType.ARRAY:
        try: 
            return field.contents() # works for GPT-2 strs
        except UnicodeDecodeError:
            return [bytes(field.parts[idx]) for idx in field.data] # works for byte obj
    
    return field.contents()

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

def handle_conversion_error(e: Exception, args) -> bool:
    """ Handles conversion errors with helpful hints. Returns True if error handled, False for re-raise. """
    error_msg = str(e)
    
    if isinstance(e, TypeError):
        if args.encode:
            print(f"Error: file appears to already be encoded type (wrong type)")
            print(f"Hint: Try using --decode (-d) instead")
        else:
            print(f"Error: file appears to already be decoded type (wrong type)")
            print(f"Hint: Try using --encode (-e) instead")
        return True
    
    elif isinstance(e, ValueError):
        if "Invalid merge format" in error_msg:
            print(f"Error: Invalid merge format in file: {error_msg}")
            return True
        elif args.encode and "not found in BYTE_ENCODER" in error_msg:
            print(f"Error: file appears to already be encoded type (contains strings, not bytes)")
            print(f"Hint: Try using --decode (-d) instead")
            return True
        elif args.decode and "not found in BYTE_DECODER" in error_msg:
            print(f"Error: file appears to already be decoded type (contains bytes, not GPT-2 strings)")
            print(f"Hint: Try using --encode (-e) instead")
            return True
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Modify GGUF file's tokenizer.ggml.tokens and .merges, conditional on tokenizer.ggml.model being 'gpt2'"
    )
    parser.add_argument("gguf_path", type=str, help="Path to the .gguf file")
    
    # req arg, options for it are mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--decode", action="store_true",
                      help="Decode tokens and merges ((GPT-2) strs -> bytes)")
    group.add_argument("-e", "--encode", action="store_true",
                      help="Encode tokens and merges (bytes -> (GPT-2) strs)")
    
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

        # process tokens (if present)
        tokens_field = reader.get_field("tokenizer.ggml.tokens")
        if tokens_field is None:
            print("'tokenizer.ggml.tokens' not found in metadata, no changes made")
            return
        
        tokens = get_tokens_or_merges(tokens_field)
        print(f"Processing {len(tokens)} tokens...")
        
        if args.encode:
            # bytes -> (GPT-2) strs
            new_tokens = [encode_token(t) for t in tokens]
        elif args.decode:
            # (GPT-2) strs -> bytes
            new_tokens = [decode_token(t) for t in tokens]
        else: # this should never happen, but just in case
            raise ValueError("Encode/decode flag not passed")

        # process merges (if present)
        merges_field = reader.get_field("tokenizer.ggml.merges")
        if merges_field is None:
            print("'tokenizer.ggml.merges' not found in metadata, no changes made")
            return
        
        merges = get_tokens_or_merges(merges_field)
        print(f"Processing {len(merges)} merges...")
        new_merges = []
        
        if args.encode:
            # bytes -> (GPT-2) strs
            for m in merges:
                parts = m.split(b" ", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid merge format: {m}")
                p1 = encode_token(parts[0])
                p2 = encode_token(parts[1])
                new_merges.append(p1 + " " + p2)
        elif args.decode:
            # (GPT-2) strs -> bytes
            for m in merges:
                parts = m.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"Invalid merge format: {m}")
                p1 = decode_token(parts[0])
                p2 = decode_token(parts[1])
                new_merges.append(p1 + b" " + p2)
        else: # this should never happen, but just in case
            raise ValueError("Encode/decode flag not passed")

        # tmp file in same dir as GGUF file
        dir_name = os.path.dirname(args.gguf_path)
        with tempfile.NamedTemporaryFile(dir=dir_name, delete=False) as temp_file:
            temp_path = temp_file.name
        
        arch_field = reader.get_field("general.architecture")
        arch = arch_field.contents() if arch_field else "none"
        writer = GGUFWriter(temp_path, arch)

        # writer auto-adds these keys, skip them
        auto_generated_keys = {
            "general.architecture",
            "GGUF.version",
            "GGUF.tensor_count",
            "GGUF.kv_count"
        }

        # add KV metadata
        for key, _ in reader.fields.items():
            if key in auto_generated_keys:
                continue
            elif key == "tokenizer.ggml.tokens":
                writer.add_token_list(new_tokens)
            elif key == "tokenizer.ggml.merges":
                writer.add_token_merges(new_merges)
            else:
                copy_metadata(writer, reader, key)
        
        # add tensors
        for tensor in reader.tensors:
            # work-around due to a bug w/ quant_shape_from_byte_shape in GGUF writer source code
            # (this preserves shape of the BF16 tensors)
            if tensor.tensor_type == GGMLQuantizationType.BF16 and tensor.data.dtype == np.uint8:
                tensor.data.dtype = np.float16

            writer.add_tensor(
                name=tensor.name,
                tensor=tensor.data,
                raw_shape=tuple(reversed(tensor.shape)), # GGUF writer reverses this
                raw_dtype=tensor.tensor_type,
            )
        
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # output filename needs suffixing
        # remove .gguf extension if applicable
        # remove existing suffix if present
        # add new suffix and restore extension

        base_path = args.gguf_path
        if base_path.endswith('.gguf'):
            base_name = base_path[:-5]
        else:
            base_name = base_path
        
        if base_name.endswith("_enc") or base_name.endswith("_dec"):
            base_name = base_name[:-4]
        
        suffix = "_enc" if args.encode else "_dec"
        output_path = base_name + suffix + ".gguf"
        
        shutil.move(temp_path, output_path)

        # remove orig file if different from output file
        if output_path != args.gguf_path and os.path.exists(args.gguf_path):
            os.remove(args.gguf_path)
        
        action = "encoded" if args.encode else "decoded"
        print(f"Successfully {action} and saved to {output_path}")
        
    # here we try to handle exceptions specific to the conversion with helpful hints
    # otherwise if the exception isn't ours, just re-raise
    # must clean up tmp file
    except Exception as e:
        handled = handle_conversion_error(e, args)

        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

        if not handled:
            raise

if __name__ == "__main__":
    BYTE_DECODER, BYTE_ENCODER = create_gpt2_byte_decoder()
    main()