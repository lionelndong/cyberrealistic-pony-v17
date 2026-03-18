"""
ComfyUI workflow generator for Replicate.
Uses lionelndong/cyberrealistic-pony-comfyui (CyberRealisticPony v1.70 baked in).
Supports LoRA loading by URL and img2img.
"""

import json


def build_txt2img_workflow(
    prompt: str,
    negative_prompt: str = "score_6, score_5, score_4, ugly, blurry, deformed, low quality, worst quality",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg: float = 7.0,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int = -1,
    lora_url: str = "",
    lora_strength: float = 0.8,
    lora_url_2: str = "",
    lora_strength_2: float = 0.8,
) -> str:
    """Build a ComfyUI API-format workflow for text-to-image generation.

    Args:
        prompt: The positive prompt. For Pony models, prefix with 'score_9, score_8_up, score_7_up,'
        negative_prompt: What to avoid
        width: Image width (default 1024)
        height: Image height (default 1024)
        steps: Number of denoising steps
        cfg: Classifier-free guidance scale
        sampler: Sampler name (dpmpp_2m, euler_ancestral, euler, dpmpp_sde, etc.)
        scheduler: Scheduler (karras, normal, simple, etc.)
        seed: Random seed (-1 for random)
        lora_url: URL to a LoRA .safetensors file (optional)
        lora_strength: LoRA strength (0.0-2.0)
        lora_url_2: URL to a second LoRA (optional)
        lora_strength_2: Second LoRA strength

    Returns:
        JSON string of the ComfyUI workflow
    """
    import random

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Add Pony score tags if not already present
    if "score_9" not in prompt:
        prompt = f"score_9, score_8_up, score_7_up, {prompt}"

    workflow = {}

    # Checkpoint loader
    workflow["4"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "cyberrealisticPony_v170.safetensors"
        }
    }

    # Track the current model and clip output
    model_source = ["4", 0]
    clip_source = ["4", 1]

    # LoRA 1 (if provided)
    if lora_url:
        workflow["10"] = {
            "class_type": "LoraLoaderFromURL",
            "inputs": {
                "model": model_source,
                "clip": clip_source,
                "url": lora_url,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            }
        }
        model_source = ["10", 0]
        clip_source = ["10", 1]

    # LoRA 2 (if provided)
    if lora_url_2:
        workflow["11"] = {
            "class_type": "LoraLoaderFromURL",
            "inputs": {
                "model": model_source,
                "clip": clip_source,
                "url": lora_url_2,
                "strength_model": lora_strength_2,
                "strength_clip": lora_strength_2,
            }
        }
        model_source = ["11", 0]
        clip_source = ["11", 1]

    # Positive prompt
    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": clip_source,
            "text": prompt
        }
    }

    # Negative prompt
    workflow["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": clip_source,
            "text": negative_prompt
        }
    }

    # Empty latent image
    workflow["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": height,
            "width": width
        }
    }

    # KSampler
    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "cfg": cfg,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": model_source,
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": sampler,
            "scheduler": scheduler,
            "seed": seed,
            "steps": steps
        }
    }

    # VAE Decode
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        }
    }

    # Save Image
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "output",
            "images": ["8", 0]
        }
    }

    return json.dumps(workflow)


def build_img2img_workflow(
    prompt: str,
    image_url: str,
    negative_prompt: str = "score_6, score_5, score_4, ugly, blurry, deformed, low quality, worst quality",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg: float = 7.0,
    denoise: float = 0.75,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int = -1,
    lora_url: str = "",
    lora_strength: float = 0.8,
) -> str:
    """Build a ComfyUI workflow for image-to-image generation."""
    import random

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    if "score_9" not in prompt:
        prompt = f"score_9, score_8_up, score_7_up, {prompt}"

    workflow = {}

    # Load checkpoint
    workflow["4"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "cyberrealisticPony_v170.safetensors"
        }
    }

    model_source = ["4", 0]
    clip_source = ["4", 1]

    # LoRA (if provided)
    if lora_url:
        workflow["10"] = {
            "class_type": "LoraLoaderFromURL",
            "inputs": {
                "model": model_source,
                "clip": clip_source,
                "url": lora_url,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            }
        }
        model_source = ["10", 0]
        clip_source = ["10", 1]

    # Load input image
    workflow["12"] = {
        "class_type": "LoadImage",
        "inputs": {
            "image": "input.png"
        }
    }

    # Encode to latent via VAE
    workflow["13"] = {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["12", 0],
            "vae": ["4", 2]
        }
    }

    # Prompts
    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": clip_source, "text": prompt}
    }
    workflow["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": clip_source, "text": negative_prompt}
    }

    # KSampler with img2img denoise
    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "cfg": cfg,
            "denoise": denoise,
            "latent_image": ["13", 0],
            "model": model_source,
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": sampler,
            "scheduler": scheduler,
            "seed": seed,
            "steps": steps
        }
    }

    # Decode & save
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "output", "images": ["8", 0]}
    }

    return json.dumps(workflow)
