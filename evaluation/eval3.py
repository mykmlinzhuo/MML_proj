#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算每张生成图与多个 reference 图的相似度（DINOv2 + CLIP-I）
方式：
  每张生成图 与 每张参考图 逐一计算 → 再取平均相似度

输出：
  image | dino_avg | clipi_avg

依赖：
  pip install timm open_clip_torch torchvision pillow tqdm
"""

import os, glob, argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import open_clip

def load_dino(device):
    model = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
    model.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    return model, transform

def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device=device
    )
    model.eval()
    return model, preprocess

def encode_image(model, img_tensor, model_type):
    with torch.no_grad():
        if model_type == 'dino':
            feat = model(img_tensor)
        elif model_type == 'clip':
            feat = model.encode_image(img_tensor)
        feat = F.normalize(feat.float(), dim=-1)
        return feat

def main(args):
    # 加载模型
    dino_model, dino_tf = load_dino(args.device)
    clip_model, clip_tf = load_clip(args.device)

    # 加载所有参考图 → 每张图保留单独的特征向量
    ref_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*")))
    if not ref_paths:
        raise RuntimeError(f"No reference images found in {args.ref_dir}")

    ref_feats_dino = []
    ref_feats_clip = []
    for p in ref_paths:
        pil = Image.open(p).convert("RGB")
        t_dino = dino_tf(pil).unsqueeze(0).to(args.device)
        t_clip = clip_tf(pil).unsqueeze(0).to(args.device)
        ref_feats_dino.append(encode_image(dino_model, t_dino, 'dino'))
        ref_feats_clip.append(encode_image(clip_model, t_clip, 'clip'))

    # 评估每张生成图像
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*")))
    if not img_paths:
        raise RuntimeError(f"No images found in {args.img_dir}")

    print(f"{'image':<30} {'dino_avg':>9} {'clipi_avg':>10}")
    print("-" * 54)

    for p in tqdm(img_paths, desc="Scoring"):
        pil = Image.open(p).convert("RGB")
        t_dino = dino_tf(pil).unsqueeze(0).to(args.device)
        t_clip = clip_tf(pil).unsqueeze(0).to(args.device)

        f_dino = encode_image(dino_model, t_dino, 'dino')
        f_clip = encode_image(clip_model, t_clip, 'clip')

        sims_dino = [(f_dino @ r.T).item() for r in ref_feats_dino]
        sims_clip = [(f_clip @ r.T).item() for r in ref_feats_clip]

        avg_dino = sum(sims_dino) / len(sims_dino)
        avg_clip = sum(sims_clip) / len(sims_clip)

        print(f"{os.path.basename(p):<30} {avg_dino:9.3f} {avg_clip:10.3f}")

# ───── CLI ───── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory of images to evaluate")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Directory containing reference images")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
