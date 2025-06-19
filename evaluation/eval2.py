#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只计算 CLIP-IQA 图像质量得分：
输出每张图的：
  image | clip_iqa
"""

import os, glob, argparse
from PIL import Image
from tqdm import tqdm
import torch
import pyiqa

def main(args):
    metric = pyiqa.create_metric('clipiqa').to(args.device).eval()
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*")))
    if not img_paths:
        raise RuntimeError(f"No images found in {args.img_dir}")

    print(f"{'image':<30} {'clip_iqa':>9}")
    print("-" * 40)

    for p in tqdm(img_paths, desc="Evaluating"):
        pil = Image.open(p).convert("RGB")
        with torch.no_grad():
            # ✅ 直接传入 PIL.Image，无需 preprocess
            q = metric(pil.to(args.device) if args.device == 'cpu' else pil).item()
        print(f"{os.path.basename(p):<30} {q:9.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Path to directory with images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    args = parser.parse_args()
    main(args)
