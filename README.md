# MML Project Initilization

## 2025.5.17
After conducting a pilot study on flow-based model inpainting, flow-based model distillation and flow-based model acceleration, I personally find these three codebases which would be helpful for our research.
- [Diffusion Based Model Inpainting](https://github.com/Sherrylone/PQDiff)
- [Flow-Based Model for Image Generation](https://github.com/willisma/SiT)
- [Flow-Based Model Distillation](https://github.com/willisma/SiT)

Maybe we could simply do A+B+C, haha.

### Environment

Refer to README_old.md for the environment setup.

### Setup

Download SiT-XL-2-256.pt and reach out to lz for the dataset.

### Tuning

```bash
torchrun --nproc_per_node=1 train_outpaint.py   --data-path ./small_dataset   --pretrained-ckpt ./pretrained_models/SiT-XL-2-256.pt --wandb 
```