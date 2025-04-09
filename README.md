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

## Installation

1. Clone or download this repository into your Houdini preferences/scripts/python_panels directory:

```bash
git clone https://github.com/YOUR_USERNAME/houdini-sdxl-panel.git
```

2. Install Python requirements into your Houdini Python environment:

```bash
pip install diffusers transformers accelerate torch torchvision safetensors huggingface_hub pynvml
```

3. (Optional) Log in to Hugging Face to access private or gated models:

```python
from huggingface_hub import login
login(token="your_huggingface_token")
```

---

## Usage

1. Launch Houdini.
2. Open a new Python Panel and select `SDXLPanel` from the interface list.
3. Enter your prompt and adjust settings such as scheduler, resolution, and base model.
4. (Optional) Select a control model and enable viewport capture.
5. (Optional) Add a LoRA model using:
   - **Repo ID** (e.g., `PvDeep/Add-Detail-XL`)
   - **Or** local `.safetensors` file
6. Click **Generate** to begin image generation.

---

## Notes

- Uses `snapshot_download()` for LoRA weights with `"*.safetensors"` pattern matching.
- LoRA weights are cached locally and reused to avoid re-downloading.
- GPU mode is default; toggle to CPU-only mode via button if needed.
- Generated material maps can be found in the render folder and auto-linked in `/mat`.

---

## Example LoRA Integration

To use a Hugging Face LoRA model:

```
PvDeep/Add-Detail-XL
```

The panel will automatically:

- Download safetensors file via `snapshot_download`
- Apply it with `pipe.load_lora_weights()`

Or browse a local file:

```
/path/to/your/LoRA/model.safetensors
```

---

## Contributing

Feel free to open issues or pull requests if you want to contribute or improve this panel.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

- [Hugging Face](https://huggingface.co/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Transformers](https://github.com/huggingface/transformers)
- [Stable Diffusion XL](https://stability.ai/)

---
