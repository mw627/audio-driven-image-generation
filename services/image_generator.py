import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os


class ImageGenerator:
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda"):
        self.device = device

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(self.device)

        #节省显存
        torch.backends.cuda.matmul.allow_tf32 = True
        self.pipe.enable_attention_slicing()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        output_path: str = "output.png",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
    ) -> Image.Image:

        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

        return image

    def generate_batch(
        self,
        prompt: str,
        negative_prompt: str = "",
        output_dir: str = "outputs",
        num_images: int = 3,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
    ) -> list:
        images_paths = []
        for i in range(num_images):
            # 每张图用不同种子，保证结果各不相同
            current_seed = None if seed is None else seed + i
            generator = None
            if current_seed is not None:
                generator = torch.Generator(self.device).manual_seed(current_seed)
                
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            ).images[0]

            output_path = os.path.join(output_dir, f"result_{i+1}.png")
            os.makedirs(output_dir, exist_ok=True)
            image.save(output_path)
            images_paths.append(output_path)

        return images_paths