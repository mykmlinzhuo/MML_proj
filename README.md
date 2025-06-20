# LoRA Fine-Tuning for Style Control

This branch focuses on fine-tuning Stable Diffusion v1.5 with LoRA to learn specific visual styles (e.g., Minecraft pixel-art, Monet painting) that are later composable via our multi-reference system.

## Dataset

Please contact the authors for access to the LoRA training datasets.

## Model

We use [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) as the base model.

- `v1-5-pruned.ckpt`: 7.7GB, includes EMA + non-EMA weights. Suitable for fine-tuning.

## Training

We adopt the `train_text_to_image_lora.py` script for LoRA training. A typical command:

```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=path_to_sd15_checkpoint \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=10 \
  --output_dir=output/lora_mc \
  --num_train_epochs=600 \
  --train_data_dir=dataset/mc
```

After training, the final LoRA weights are stored in `pytorch_lora_weights.bin`.



## Next Steps

Once LoRA weights are trained, please switch to the `master` branch to integrate them into our multi-concept generation pipeline.

## References

1. [Stable Diffusion 微调及推理优化](https://cloud.tencent.com/developer/article/2302436)  
2. [Using LoRA for Efficient Stable Diffusion Fine-Tuning](https://huggingface.co/blog/lora)  
3. [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)