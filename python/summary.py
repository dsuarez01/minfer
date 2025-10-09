import argparse
import os

from gguf import GGUFReader

def main():
    parser = argparse.ArgumentParser(
        description="Reads GGUF file and prints summary of tensor metadata (skips tokenizer.ggml.tokens and tokenizer.ggml.merges)"
    )
    parser.add_argument("gguf_path", type=str, help="Path to the .gguf file")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {args.gguf_path}")
        reader = GGUFReader(args.gguf_path)

        print(f"GGUF file: {args.gguf_path}")
        
        # KV metadata
        print("\n" + "=" * 50)
        print("KV Metadata:")
        print("=" * 50)
        skip_keys = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}
        for key, field in reader.fields.items():
            if key in skip_keys:
                print(f"{key}: <skipped>")
                continue
            try:
                value = field.contents()
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    # truncation for readability
                    print(f"{key}: [{value[0]}, {value[1]}, ..., {value[-1]}] (length: {len(value)})")
                else:
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"{key}: <error reading: {e}>")
        
        # tensor metadata
        print("\n" + "=" * 50)
        print(f"Total tensors: {len(reader.tensors)}")
        print("=" * 50)
        print("\nTensor Metadata:")
        print("-" * 50)
        for tensor in reader.tensors:
            dtype_size = {
                'F32': 4, 'F16': 2, 'BF16': 2,
            }[tensor.tensor_type.name]
            shape_product = 1
            for dim in tensor.shape:
                shape_product *= dim
            size_bytes = shape_product * dtype_size

            print(f"Name: {tensor.name}")
            print(f"  Shape: {list(tensor.shape)}")
            print(f"  Dtype: {tensor.tensor_type.name}")
            print(f"  Numpy dtype: {tensor.data.dtype}")
            print(f"  Size: {size_bytes:,} bytes")
            print("-" * 50)

    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()