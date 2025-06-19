from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("/cephfs/shared/linzhuo/multimodal/FreeCustom/sd_v1.5", subfolder="unet")
unet.load_attn_procs("/cephfs/shared/linzhuo/multimodal/stable-diffusion-v1.5-lora/output/mc")
print("✅ LoRA loaded successfully!")
