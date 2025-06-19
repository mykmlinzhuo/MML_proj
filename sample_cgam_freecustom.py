import os, sys, datetime
from omegaconf import OmegaConf

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from diffusers import DDIMScheduler

from utils.utils import load_image, load_mask
from pipelines.pipeline_stable_diffusion_freecustom import StableDiffusionFreeCustomPipeline
from freecustom.mrsa import MultiReferenceSelfAttention
from freecustom.hack_attention import hack_self_attention_to_mrsa
# >>> 新增：CGAM
from freecustom.cgam import CGAMController

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # -------------------------------------------------------
    # basic config & dirs
    # -------------------------------------------------------
    cfg = OmegaConf.load("configs/config_stable_diffusion.yaml")
    print(f'config: {cfg}')

    date = datetime.datetime.now().strftime("%Y%m%d")
    now  = datetime.datetime.now().strftime("%H-%M-%S")
    results_root = os.path.join('results', date, now)

    torch.cuda.set_device(cfg.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -------------------------------------------------------
    # load pipeline
    # -------------------------------------------------------
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False
    )
    model = StableDiffusionFreeCustomPipeline.from_pretrained(
        cfg.model_path, scheduler=scheduler
    ).to(device)
    model.safety_checker = None

    # 可选：LoRA
    if getattr(cfg, "lora_path", None):
        model.unet.load_attn_procs(cfg.lora_path)
        print(f">> LoRA loaded from: {cfg.lora_path}")

    # -------------------------------------------------------
    # prepare reference data
    # -------------------------------------------------------
    ref_masks, ref_images, ref_prompts, ref_latents_z_0 = [], [], [], []
    for ref_image_path, ref_text_prompt in cfg.ref_image_infos.items():
        ref_mask_path = ref_image_path.replace('/image/', '/mask/')
        ref_masks.append(load_mask(ref_mask_path, device))
        ref_img = load_image(ref_image_path, device)
        ref_images.append(ref_img)
        ref_prompts.append(ref_text_prompt)
        ref_latents_z_0.append(model.image2latent(ref_img))

    # prompt 设置
    target_prompt = cfg.target_prompt
    prompts = [target_prompt] + ([""] * len(ref_prompts)) if cfg.use_null_ref_prompts else [target_prompt] + ref_prompts
    negative_prompts = [cfg.negative_prompt] * len(prompts)

    # 结果目录
    concepts_name = list(cfg.ref_image_infos.keys())[0].split('/')[3]
    run_dir = os.path.join(results_root, f"{concepts_name} \"{target_prompt}\" {cfg.mark}")
    img_dir, msk_dir = os.path.join(run_dir, 'ref_images'), os.path.join(run_dir, 'ref_masks')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(msk_dir,  exist_ok=True)

    # 可视化 config
    viz_cfg = OmegaConf.load("configs/config_for_visualization.yaml")
    viz_cfg.results_dir = run_dir
    viz_cfg.ref_image_infos = cfg.ref_image_infos
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    for i, (im, mk) in enumerate(zip(ref_images, ref_masks)):
        save_image(im * 0.5 + 0.5, os.path.join(img_dir, f'image_{i}.png'))
        save_image(mk.float(),            os.path.join(msk_dir, f'mask_{i}.png'))

    # -------------------------------------------------------
    # run seeds
    # -------------------------------------------------------
    for seed in cfg.seeds:
        seed_everything(seed)

        # ------ patch MRSA ------
        mrsa = MultiReferenceSelfAttention(
            start_step     = cfg.start_step,
            end_step       = cfg.end_step,
            layer_idx      = cfg.layer_idx,
            ref_masks      = ref_masks,
            mask_weights   = cfg.mask_weights,
            style_fidelity = cfg.style_fidelity,
            viz_cfg        = viz_cfg
        )
        hack_self_attention_to_mrsa(model, mrsa)

        # ------ NEW: CGAM controller ------
        cgam = CGAMController(
            ref_images = ref_images,
            vae        = model.vae,
            mrsa_layers= [mrsa],        # 如果 hack 出多个 MRSA 实例，替换为列表
            device     = device,
            k          = cfg.get("cgam_k", 8),
            tau        = cfg.get("cgam_tau", 0.07),
        )

        # monkey-patch scheduler.step 以注入 CGAM
        orig_step = model.scheduler.step
        def step_with_cgam(*args, **kwargs):
            out = orig_step(*args, **kwargs)

            # handle both return_dict=False (tuple) and True (dict-like)
            if isinstance(out, tuple):
                z_t = out[0]  # this is the updated latent
            else:
                z_t = getattr(out, "prev_sample", out.sample)

            cgam.update(z_t)
            return out
        model.scheduler.step = step_with_cgam
        # ----------------------------------

        # latent 初始化
        randn_latent_z_T = torch.randn_like(ref_latents_z_0[0])
        latents = torch.cat([randn_latent_z_T] + ref_latents_z_0)

        # 采样
        gen_img = model(
            prompt=prompts,
            latents=latents,
            guidance_scale=7.5,
            negative_prompt=negative_prompts,
        ).images[0]

        # 保存
        save_path = os.path.join(run_dir, f"freecustom_cgam_{seed}.png")
        gen_img.save(save_path)
        print(f">> saved: {save_path}")

        # 拼接展示
        grid = torch.cat(
            [im * 0.5 + 0.5 for im in ref_images] +
            [ToTensor()(gen_img).to(device).unsqueeze(0)],
            dim=0
        )
        save_image(grid, os.path.join(run_dir, f"all_{seed}.png"))
