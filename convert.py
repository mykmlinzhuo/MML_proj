import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# 加载 base 模型（SD v1.5）
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/cephfs/shared/linzhuo/multimodal/FreeCustom/sd_v1.5", 
    torch_dtype=torch.float16
).to("cuda")
pipe.safety_checker = None  # 可选：关闭 NSFW 检查

# 加载参考图像（作为 style）和输入图（content）
init_image = Image.open("freecustom_2.png").convert("RGB").resize((512, 512))

# prompt 中加入你希望的风格描述，比如“in Van Gogh style”
prompt = "Wikiart style, in the style of Van Gogh, vibrant colors, swirling patterns, "  
negative_prompt = "blurry, distorted, low quality, "

# 推理（strength 控制保留原图比例）
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.3,            # [0, 1]，越大越偏向生成图风格
    guidance_scale=4,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
).images[0]

# 保存结果
os.makedirs("outputs", exist_ok=True)
image.save("outputs/stylized_result.png")
print("✅ done.")
