# CyberRealisticPony v17 on Replicate

CyberRealisticPony v1.70 (SDXL-based, from CivitAI) packaged with [Cog](https://github.com/replicate/cog) for deployment on [Replicate](https://replicate.com).

## Features

- **CyberRealisticPony v1.70** checkpoint baked into the image
- **LoRA support** — load up to 2 LoRAs from URL at inference time
- **Textual inversion embeddings** — load up to 2 embeddings from URL
- **img2img** — pass an input image + strength
- **9 schedulers** — DPM++ 2M Karras, Euler a, Heun, UniPC, etc.
- **Batch generation** — up to 4 images per request

## Deploy to Replicate

### Prerequisites

1. Install [Cog](https://github.com/replicate/cog#install)
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
3. Create a Replicate account at https://replicate.com
4. Log in:

```bash
cog login
```

### Push the model

```bash
# 1. Create your model at https://replicate.com/create

# 2. Push (first push bundles the 6.5GB checkpoint — takes a while)
cog push r8.im/<your-username>/<your-model-name>
```

## Usage

```python
import replicate

output = replicate.run(
    "<your-username>/<your-model-name>",
    input={
        "prompt": "a beautiful landscape, photorealistic",
        "negative_prompt": "ugly, blurry",
        "lora_url": "https://example.com/my-lora.safetensors",
        "lora_scale": 0.8,
        "scheduler": "DPM++ 2M Karras",
        "num_inference_steps": 30,
        "width": 1024,
        "height": 1024,
    }
)
```
