import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from copy import deepcopy
from models import SiT_models
from dataset.dataset import get_dataset
from diffusers.models import AutoencoderKL
from time import time
from collections import OrderedDict
import wandb


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def freeze_all_but_condition(model):
    for name, param in model.named_parameters():
        param.requires_grad = ("cond_embedder" in name)


def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    local_bs = args.global_batch_size // dist.get_world_size()

    if rank == 0 and args.wandb:
        wandb.init(project="sit-outpaint", config=vars(args))

    latent_size = args.image_size // 8
    model = SiT_models[args.model](input_size=latent_size)
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    # model.load_state_dict(ckpt["model"], strict=False)
    model.load_state_dict(ckpt, strict=False)
    freeze_all_but_condition(model)

    model = DDP(model.to(device), device_ids=[rank])
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    vae = AutoencoderKL.from_pretrained("./vae").to(device)

    dataset = get_dataset(
        name="scenery",
        path=args.data_path,
        resolution=args.image_size,
        embed_dim=320,
        grid_size=16,
    )
    loader = DataLoader(dataset.get_split("train", labeled=True), batch_size=local_bs, shuffle=True, num_workers=4, drop_last=True)

    model.train()
    ema.eval()
    step = 0
    for epoch in range(args.epochs):
        for target_img, anchor_img, rpe_token in loader:
            target_img = target_img.to(device).float()
            anchor_img = anchor_img.to(device).float()
            rpe_token = rpe_token.to(device).float()

            with torch.no_grad():
                z_target = vae.encode(target_img).latent_dist.sample().mul_(0.18215)
                z_anchor = vae.encode(anchor_img).latent_dist.sample().mul_(0.18215)

            t = torch.rand(z_target.size(0), device=device)
            z_t = z_target * (1 - t[:, None, None, None]) + torch.randn_like(z_target) * t[:, None, None, None]

            anchor_token = model.module.x_embedder(z_anchor)  # 注意要加 .module
            pred_v = model(z_t, t, anchor_token=anchor_token, rpe_token=rpe_token)
            true_v = (z_target - z_t) / t[:, None, None, None]  # simple linear ODE velocity

            loss = F.mse_loss(pred_v, true_v)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema.module, model.module)

            if step % 10 == 0 and rank == 0:
                print(f"step {step} | loss {loss.item():.4f}")
                if args.wandb:
                    wandb.log({"loss": loss.item(), "step": step})

            if step % args.save_every == 0 and rank == 0 and step > 0:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "args": vars(args)
                }, os.path.join(args.save_dir, f"step_{step:06d}.pt"))
                print(f"[rank {rank}] Saved checkpoint at step {step}")

            step += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--pretrained-ckpt", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_outpaint")
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()
    train(args)
