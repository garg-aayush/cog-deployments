import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image


# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "black-forest-labs/FLUX.1-dev"
CONTROLNET_ID = "jasperai/Flux.1-dev-Controlnet-Upscaler"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the pipe and controlnet into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        # Load pipeline
        controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image",
        ),
        scale_factor: float = Input(
            description="Scale factor for image upscaling", ge=1, le=5, default=4
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=28
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale for controlnet conditioning", ge=0.1, le=1.5, default=0.6
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0.5, le=5, default=3.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)
       
        # Load a control image
        control_image = Image.open(image)
        w, h = control_image.size

        # Upscale x4
        control_image = control_image.resize((int(w * scale_factor), int(h * scale_factor)))

        image = self.pipe(
            prompt="", 
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            height=control_image.size[1],
            width=control_image.size[0]
        ).images[0]

        
        output_path = Path("/tmp/out.png")
        image.save(output_path)
        
        return [output_path]
