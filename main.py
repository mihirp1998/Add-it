import torch
from diffusers import FluxPipeline, FluxInpaintPipeline, FluxImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import load_image
from PIL import Image
import builtins
import numpy as np
import ipdb
st = ipdb.set_trace
builtins.st = st
import gc
import os
import glob


height = 512
width = 512
multi_step = 670
single_step = 340
gamma = 1.05
strength = 0.87
add_it = True
guidance_scale = 7.0
num_inference_steps = 34


pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")
vae_scale_factor = pipe.vae_scale_factor
in_channels = pipe.transformer.config.in_channels

prompts = ["A horse standing in the garden.", "The horse is wearing a pink dress."]
initial_prompt = prompts[0]



tmp_height = 2 * (int(height) // (vae_scale_factor * 2))
tmp_width = 2 * (int(width) // (vae_scale_factor * 2))
num_channels_latents = in_channels // 4
shape = (1, num_channels_latents, tmp_height, tmp_width)
noise_init = randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(0), device=torch.device("cuda"), dtype=torch.bfloat16)
noise_init_packed = pipe._pack_latents(noise_init, 1, num_channels_latents, tmp_height, tmp_width)

init_image = pipe(
    initial_prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    latents=noise_init_packed
).images[0]

os.makedirs("results", exist_ok=True)

init_image.save("results/input.png")

del pipe
torch.cuda.empty_cache()
gc.collect()


img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    num_inference_steps=num_inference_steps, 
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True
)
img2img_pipe.to("cuda")


prompt = initial_prompt
image_to_edit = init_image
previous_prompts = [initial_prompt]



for i, new_prompt in enumerate(prompts[1:]):
    prompt = " ".join(previous_prompts)
    prompts = [prompt, new_prompt]
    print(prompts)
    flux_out, target_noise = img2img_pipe(prompts, num_inference_steps=num_inference_steps,  height=height, width=width, image=image_to_edit, guidance_scale=guidance_scale, strength=strength, gamma=gamma,  add_it=add_it, source_noise=noise_init)
    images = flux_out.images
    
    noise_init = target_noise
    images[1].save(f"final/generated_edited_{i+1}.png")
    print(f"Generated image {i+1}")
    image_to_edit = images[1]
    previous_prompts.append(new_prompt)
