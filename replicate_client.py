"""
Replicate client for generating images via ComfyUI.
Uses CyberRealisticPony v1.70 on ComfyUI via lionelndong/cyberrealistic-pony-comfyui.

Usage:
    from replicate_client import generate_image

    # Simple text-to-image
    images = generate_image(
        prompt="a beautiful portrait, photorealistic",
        api_token="r8_your_token_here"
    )

    # With LoRA
    images = generate_image(
        prompt="a woman in cyberpunk style",
        lora_url="https://example.com/my-lora.safetensors",
        lora_strength=0.8,
        api_token="r8_your_token_here"
    )
"""

import json
import time
import requests
from comfyui_workflows import build_txt2img_workflow, build_img2img_workflow

# CyberRealisticPony v1.70 on ComfyUI
COMFYUI_VERSION = "7568d4a6e4471e7aa3cbcda25986257a1b7c5a5fc333f74a1bdc2399c27a440d"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"


def generate_image(
    prompt: str,
    api_token: str,
    negative_prompt: str = "score_6, score_5, score_4, ugly, blurry, deformed, low quality",
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
    timeout: int = 120,
) -> list[str]:
    """Generate images using ComfyUI on Replicate.

    Returns a list of image URLs.
    """
    workflow_json = build_txt2img_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        lora_url=lora_url,
        lora_strength=lora_strength,
        lora_url_2=lora_url_2,
        lora_strength_2=lora_strength_2,
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # Start prediction
    response = requests.post(
        REPLICATE_API_URL,
        headers=headers,
        json={
            "version": COMFYUI_VERSION,
            "input": {"workflow_json": workflow_json},
        },
        timeout=30,
    )
    response.raise_for_status()
    prediction = response.json()
    prediction_id = prediction["id"]

    # Poll for result
    poll_url = f"{REPLICATE_API_URL}/{prediction_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(2)
        result = requests.get(poll_url, headers=headers, timeout=15).json()
        status = result.get("status")

        if status == "succeeded":
            return result.get("output", [])
        elif status in ("failed", "canceled"):
            raise RuntimeError(
                f"Prediction {status}: {result.get('error', 'unknown error')}"
            )

    raise TimeoutError(f"Prediction {prediction_id} did not complete in {timeout}s")


def generate_img2img(
    prompt: str,
    image_url: str,
    api_token: str,
    negative_prompt: str = "score_6, score_5, score_4, ugly, blurry, deformed",
    denoise: float = 0.75,
    steps: int = 25,
    cfg: float = 7.0,
    seed: int = -1,
    lora_url: str = "",
    lora_strength: float = 0.8,
    timeout: int = 120,
) -> list[str]:
    """Generate images using img2img via ComfyUI."""
    workflow_json = build_img2img_workflow(
        prompt=prompt,
        image_url=image_url,
        negative_prompt=negative_prompt,
        denoise=denoise,
        steps=steps,
        cfg=cfg,
        seed=seed,
        lora_url=lora_url,
        lora_strength=lora_strength,
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        REPLICATE_API_URL,
        headers=headers,
        json={
            "version": COMFYUI_VERSION,
            "input": {
                "workflow_json": workflow_json,
                "input_file": image_url,
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    prediction = response.json()
    prediction_id = prediction["id"]

    poll_url = f"{REPLICATE_API_URL}/{prediction_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(2)
        result = requests.get(poll_url, headers=headers, timeout=15).json()
        status = result.get("status")

        if status == "succeeded":
            return result.get("output", [])
        elif status in ("failed", "canceled"):
            raise RuntimeError(
                f"Prediction {status}: {result.get('error', 'unknown error')}"
            )

    raise TimeoutError(f"Prediction {prediction_id} did not complete in {timeout}s")


if __name__ == "__main__":
    import sys

    token = sys.argv[1] if len(sys.argv) > 1 else input("Replicate API token: ")
    prompt = sys.argv[2] if len(sys.argv) > 2 else "a beautiful woman, photorealistic, soft lighting"

    print(f"Generating: {prompt}")
    images = generate_image(prompt=prompt, api_token=token)
    for i, url in enumerate(images):
        print(f"Image {i+1}: {url}")
