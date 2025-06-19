#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_clip_concepts.py
------------------------------------------------------------
对每张图评估四个概念的匹配度，并打印 7 列结果：
  image | style | dog | sunglasses | hat | overall_avg | content_avg
依赖：
  pip install open_clip_torch tqdm pillow torch torchvision
用法：
  python eval_clip_concepts.py --img_dir ./generated --device cuda
------------------------------------------------------------
"""

import os, glob, argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import open_clip

# ────────────────────────────── 配置 ────────────────────────────── #
CONCEPTS = {
    "style":       "a Minecraft style pixelated image",
    "dog":         "a dog",
    "sunglasses":  "wearing sunglasses",
    "hat":         "wearing a red and black hat",
}

MODEL_NAME = "ViT-B-32"
PRETRAINED  = "openai"  # 可选: "laion400m_e32", "laion2b_s34b_b79k", "openai"

# ──────────────────────────── 主函数 ───────────────────────────── #
def main(args):
    # 1) 加载模型 + 预处理
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=args.device
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    # 2) 编码文本概念向量
    text_feats = {}
    for cname, prompt in CONCEPTS.items():
        txt = tokenizer([prompt]).to(args.device)
        with torch.no_grad():
            feat = model.encode_text(txt).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
        text_feats[cname] = feat

    # 3) 遍历图片
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*")))
    if not img_paths:
        raise RuntimeError(f"No images found in {args.img_dir}")

    header = (
        f"{'image':<25} {'style':>7} {'dog':>7} {'sungl.':>7} {'hat':>7}"
        f" {'overall':>8} {'content':>8}"
    )
    print(header)
    print("-" * len(header))

    for p in tqdm(img_paths, desc="Evaluating"):
        pil = Image.open(p).convert("RGB")
        img = preprocess(pil).unsqueeze(0).to(args.device)
        with torch.no_grad():
            img_feat = model.encode_image(img).float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # 4) 计算各概念分数
        scores = {}
        for cname, tfeat in text_feats.items():
            scores[cname] = (img_feat @ tfeat.T).item()

        overall_avg  = np.mean(list(scores.values()))
        content_avg  = np.mean([scores["dog"], scores["sunglasses"], scores["hat"]])

        # 5) 打印结果
        print(
            f"{os.path.basename(p):<25}"
            f"{scores['style']:7.3f}{scores['dog']:7.3f}"
            f"{scores['sunglasses']:7.3f}{scores['hat']:7.3f}"
            f"{overall_avg:8.3f}{content_avg:8.3f}"
        )

# ───────────────────────── CLI ───────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="directory containing images to evaluate")
    parser.add_argument("--device",  type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)
