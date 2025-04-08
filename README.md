# Houdini SDXL Panel

A powerful generative AI panel integrated directly into Houdini, enabling seamless image generation using Stable Diffusion XL (SDXL), with advanced support for LoRA models, control adapters, image-to-image workflows, and more.

## Features

- 💡 **Prompt-to-Image**: Generate high-quality images using SDXL base models and custom prompts.
- 🖼 **Image-to-Image (i2i)**: Capture Houdini viewport and guide image generation with control models.
- 🔌 **LoRA Model Integration**:
  - Accepts either HuggingFace repo ID (e.g., `PvDeep/Add-Detail-XL`) or a local `.safetensors` file.
  - Automatically downloads LoRA weights using `snapshot_download()` with pattern matching.
- 🧠 **Control Adapters (T2I-Adapters)**:
  - Includes support for depth, canny, sketch, segmentation, and lineart-based conditioning.
- 🛠 **Optimized Pipeline**:
  - Supports `xformers`, memory-efficient attention, VAE tiling, and CPU/GPU offload.
- 📈 **Live GPU Monitoring**: Real-time display of GPU memory and usage via `pynvml`.
- 📸 **Captioning**: Generate a prompt using a ViT-GPT2 caption model based on Houdini's current viewport render.
- 🧾 **Presets & History**: Save/load prompt and render settings. Automatically store generation history.
- 🧪 **Material Maps Generation**: Optionally generate bump, specular, roughness, and displacement maps.
- 🧵 **Renderer Integration**: Automatically applies generated textures to a Principled Shader in `/mat`.

---
