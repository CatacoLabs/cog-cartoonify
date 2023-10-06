from cog import BasePredictor, Input, Path
import os
import sys
import torch
from PIL import Image
from clip_interrogator import Interrogator, Config
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline
)

sys.path.append('/root/blip')

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_MODEL_CACHE = "sdxl-cache/"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        print("Loading CLIP pipeline...")
        self.ci = Interrogator(
            Config(
                clip_model_name="ViT-H-14/laion2b_s32b_b79k",
                clip_model_path='cache',
                device='cuda:0',
            )
        )
        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")

    # Remove first part of prompt
    def remove_first_part(self, input_string: str):
        parts = input_string.split(',', 1)
        if len(parts) > 1:
            return parts[1].strip()
        else:
            return input_string.strip()
        
    def predict(
        self,
        image: Path = Input(description="Input image"),
        # prompt_strength: float = Input(
        #     description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        #     ge=0.0,
        #     le=1.0,
        #     default=0.8,
        # ),
        # prompt: str = Input(description="Prompt text", default=None),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Hardcoded params
        width=1024
        height=1024
        scheduler="K_EULER_ANCESTRAL"

        # Run CLIP
        image = Image.open(str(image)).convert("RGB")
        # Resize input image to SDXL size
        image = image.resize((1024, 1024))
        clip_txt = self.ci.interrogate(image)
        # print("CLIP OG:"+clip_txt)
        # clip_txt = self.remove_first_part(clip_txt)
        # print("CLIP Filtered:"+clip_txt)

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        sdxl_kwargs = {}
        # print("img2img mode")
        sdxl_kwargs["image"] = image
        sdxl_kwargs["strength"] = 0.9
        pipe = self.img2img_pipe
        
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        full_prompt = "A cartoon portait picture, full art illustration, "
        #Check if user added prompt
        # if prompt is not None:
        #     filtered_prompt = prompt.replace(")", "")
        #     full_prompt += filtered_prompt+", "
        full_prompt += clip_txt
        print("Final prompt: " + full_prompt)

        common_args = {
            "prompt": full_prompt,
            "negative_prompt": "",
            "guidance_scale": 7.5,
            "generator": generator,
            "num_inference_steps": 40,
        }
        output = pipe(**common_args, **sdxl_kwargs)

        output_path = f"/tmp/output.png"
        img_out = output.images[0]
        img_out.save(output_path)

        return Path(output_path)
        # return clip_txt