# 🎨 CLIP the Concept, Diffuse the Style

**Official implementation of our MML Project**:  

> *CLIP the Concept, Diffuse the Style: Towards Style-Consistent and Concept-Aware Diffusion Generation*  

---

This repository presents a **lightweight tuning framework** for generating visually coherent images under hybrid visual guidance—including both high-level semantic concepts and low-level artistic styles. Our method extends the FreeCustom baseline by introducing:

- 🧠 **LoRA-based style modulation**  
- 🔍 **Multi-Reference Self-Attention (MRSA)** for concept fusion  
- 🎯 **CLIP-Guided Adaptive Masking (CGAM)** for dynamic reference weighting  
- 🔄 **Soft Gated Attention** for smooth feature integration

---

## 🌿 Repository Structure

- **`master` branch**  
  Contains our **main pipeline** for multi-concept + style-controlled image generation. You can directly run inference and reproduce the results in the paper.
- **`lora` branch**  
  Provides **training scripts for LoRA-style modules**, used to model specific target styles (e.g., Minecraft pixel art, Monet impressionism) using 20-shot reference images per style.

