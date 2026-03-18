"""CyberRealisticPony v17 predictor with LoRA and embedding support for Replicate."""

import os
import tempfile
import requests
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

HF_REPO_ID = "EternalLeo/cyberrealistic-pony-v170"
HF_FILENAME = "cyberrealisticPony_v170.safetensors"
MODEL_CACHE = "/src/model-cache"

SCHEDULERS = {
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True),
    "DPM++ 2M SDE Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "Heun": HeunDiscreteScheduler,
    "KDPM2": KDPM2DiscreteScheduler,
    "KDPM2 a": KDPM2AncestralDiscreteScheduler,
    "UniPC": UniPCMultistepScheduler,
}


def download_file(url: str, dest: str) -> str:
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


class Predictor(BasePredictor):
    def setup(self):
        """Download and load the CyberRealisticPony checkpoint."""
        print("Downloading CyberRealisticPony v1.70 from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=MODEL_CACHE,
        )
        print(f"Checkpoint downloaded to: {checkpoint_path}")

        print("Loading pipeline from single-file checkpoint...")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        # Share components for img2img pipeline (saves VRAM)
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()
        self.img2img_pipe.enable_xformers_memory_efficient_attention()
        print("Model loaded successfully.")

    def _load_lora(self, pipe, lora_url: str, lora_scale: float):
        """Load a LoRA from a URL (.safetensors)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora.safetensors")
            if lora_url.startswith("http"):
                print(f"Downloading LoRA from {lora_url}...")
                download_file(lora_url, lora_path)
            else:
                lora_path = lora_url

            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_scale)

    def _unload_lora(self, pipe):
        """Unload any previously fused LoRA weights."""
        try:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        except Exception:
            pass

    def _load_embedding(self, pipe, embedding_url: str, token: str):
        """Load a textual inversion embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = ".safetensors" if "safetensors" in embedding_url else ".bin"
            emb_path = os.path.join(tmpdir, f"embedding{ext}")
            if embedding_url.startswith("http"):
                print(f"Downloading embedding from {embedding_url}...")
                download_file(embedding_url, emb_path)
            else:
                emb_path = embedding_url

            pipe.load_textual_inversion(emb_path, token=token)

    def _set_scheduler(self, pipe, scheduler_name: str):
        """Set the scheduler/sampler."""
        if scheduler_name in SCHEDULERS:
            sched = SCHEDULERS[scheduler_name]
            if callable(sched) and not isinstance(sched, type):
                pipe.scheduler = sched(pipe.scheduler.config)
            else:
                pipe.scheduler = sched.from_config(pipe.scheduler.config)

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        negative_prompt: str = Input(
            description="Negative prompt — what to avoid in the image",
            default="ugly, blurry, low quality, distorted, deformed",
        ),
        image: Path = Input(
            description="Input image for img2img mode (optional). Leave empty for txt2img.",
            default=None,
        ),
        width: int = Input(
            description="Image width", default=1024, ge=512, le=2048
        ),
        height: int = Input(
            description="Image height", default=1024, ge=512, le=2048
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=30, ge=1, le=100
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale", default=7.0, ge=0, le=20
        ),
        strength: float = Input(
            description="Denoising strength for img2img (0.0 = no change, 1.0 = full denoise)",
            default=0.75,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="Random seed. Set to -1 for random.", default=-1
        ),
        scheduler: str = Input(
            description="Sampler/scheduler to use",
            default="DPM++ 2M Karras",
            choices=[
                "DPM++ 2M",
                "DPM++ 2M Karras",
                "DPM++ 2M SDE Karras",
                "Euler",
                "Euler a",
                "Heun",
                "KDPM2",
                "KDPM2 a",
                "UniPC",
            ],
        ),
        num_outputs: int = Input(
            description="Number of images to generate", default=1, ge=1, le=4
        ),
        lora_url: str = Input(
            description="URL to a LoRA weights file (.safetensors). Leave empty for none.",
            default="",
        ),
        lora_scale: float = Input(
            description="LoRA influence strength",
            default=0.8,
            ge=0.0,
            le=2.0,
        ),
        lora_url_2: str = Input(
            description="URL to a second LoRA weights file (optional)",
            default="",
        ),
        lora_scale_2: float = Input(
            description="Second LoRA influence strength",
            default=0.8,
            ge=0.0,
            le=2.0,
        ),
        embedding_url: str = Input(
            description="URL to a textual inversion embedding file (.safetensors or .bin). Leave empty for none.",
            default="",
        ),
        embedding_token: str = Input(
            description="Trigger token for the embedding (e.g. '<my-style>')",
            default="<embedding>",
        ),
        embedding_url_2: str = Input(
            description="URL to a second textual inversion embedding (optional)",
            default="",
        ),
        embedding_token_2: str = Input(
            description="Trigger token for the second embedding",
            default="<embedding2>",
        ),
    ) -> list[Path]:
        """Run inference and return generated images."""

        # Determine which pipeline to use
        is_img2img = image is not None
        pipe = self.img2img_pipe if is_img2img else self.pipe

        # Reset previous LoRAs
        self._unload_lora(pipe)

        # Set scheduler
        self._set_scheduler(pipe, scheduler)

        # Load LoRAs
        if lora_url:
            self._load_lora(pipe, lora_url, lora_scale)

        if lora_url_2:
            self._load_lora(pipe, lora_url_2, lora_scale_2)

        # Load embeddings
        if embedding_url:
            self._load_embedding(pipe, embedding_url, embedding_token)

        if embedding_url_2:
            self._load_embedding(pipe, embedding_url_2, embedding_token_2)

        # Seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        # Build generation kwargs
        common_kwargs = dict(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        if is_img2img:
            init_image = load_image(str(image)).resize((width, height))
            common_kwargs["image"] = init_image
            common_kwargs["strength"] = strength
        else:
            common_kwargs["width"] = width
            common_kwargs["height"] = height

        # Generate
        result = pipe(**common_kwargs)

        # Save outputs
        output_paths = []
        for i, img in enumerate(result.images):
            out_path = f"/tmp/output_{i}.png"
            img.save(out_path)
            output_paths.append(Path(out_path))

        # Cleanup LoRAs for next run
        self._unload_lora(pipe)

        return output_paths
