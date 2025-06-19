import os
import torch
from safetensors.torch import load_file

def convert_safetensors_to_bin(safetensors_path, output_path=None):
    assert safetensors_path.endswith(".safetensors"), "Input must be a .safetensors file"

    # 默认输出路径
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(safetensors_path),
            "pytorch_lora_weights.bin"
        )

    # 加载 .safetensors 文件
    state_dict = load_file(safetensors_path)

    # 保存为 .bin 文件
    torch.save(state_dict, output_path)
    print(f"✅ Converted to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to .safetensors file")
    parser.add_argument("--dst", type=str, default=None, help="Path to save .bin file (optional)")
    args = parser.parse_args()

    convert_safetensors_to_bin(args.src, args.dst)
