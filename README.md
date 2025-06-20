# Multi-Concept Image Generation (Master Branch)
 This branch implements our main pipeline for **style-consistent and concept-aware image synthesis** based on the report *CLIP the Concept, Diffuse the Style*. 
 It builds upon the [FreeCustom](https://github.com/aim-uofa/FreeCustom) framework and introduces: 
 - **LoRA-based style modulation** for precise appearance control. 
 -  **Multi-Reference Self-Attention (MRSA)** for fusing multiple concept images. 
 - **CLIP-Guided Adaptive Masking (CGAM)** for dynamic conflict resolution. Our method is tuning-free for multi-concept composition, and supports plug-and-play deployment with diffusion models. 
  ## 🚀 Quick Start 
  Install dependencies: 
  ```bash 
  conda create -n mcdiff python=3.10 -y 
  conda activate mcdiff 
  pip install -r requirements.txt 
  ```
  ### 🔧 Run Generation 
  Edit the configuration file: 
  ```bash 
  configs/config_stable_diffusion.yaml
  ``` 
  Specify: 
  - `ref_image_infos`: paths to multiple concept reference images and masks. 
- `target_prompt`: text prompt that fuses the concepts. - `output`: output directory and naming. 
- (Optional) `mask_weights`, `guidance_scale`, `style_adapter` etc. Then run: 
```bash 
python freecustom_stable_diffusion.py 
``` 
and
```bash
python sample_cgam_freecustom.py
```
The generated results will appear in the specified output folder.
## 📊 Evaluation 
To reproduce our evaluation, refer to the `evaluation/` directory for CLIP-based similarity scoring and user study scripts. 
## 🙏 Acknowledgements 
This project is adapted from [FreeCustom](https://github.com/aim-uofa/FreeCustom), with additional modules for adaptive multi-concept fusion. We also thank: - [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) - [HuggingFace diffusers](https://github.com/huggingface/diffusers) 