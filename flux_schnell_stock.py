# file: flux_schnell_stock.py

import time, os, torch, random
from diffusers import FluxPipeline

print("Is cuda available: ", torch.cuda.is_available())
print("CUDA device name: ", torch.cuda.get_device_name(0))

# allocator knobs
if os.name == "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=torch.bfloat16
)
# offload everything except the hot path; VAE tiling to keep memory sane
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_attention_slicing()

pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

def generate_image(prompt, image_size):

    num_timesteps = 3
    random_seed = random.randint(0, 2**32 - 1)  # 32-bit unsigned int range

    if image_size is None:
        image_size = (256,256)

    width, height = image_size

    image_pil = pipe(
        prompt,
        guidance_scale=0.0,
        height=height,
        width=width,
        num_inference_steps=num_timesteps,
        max_sequence_length=256
    #    generator=torch.Generator(device="cuda").manual_seed(random_seed)
    ).images[0]

    return image_pil

#
# MAIN 
#
if __name__ == "__main__":

    time_start = time.perf_counter()  # High‑resolution timer

    image_pil = generate_image(
        prompt="You have entered a dark crypt made of crumbling stone.  Distant echoes of collapsing piles of rubble and bones disturb the otherwise deathly silence.  Strange diagrams and ancient runes decorate the odd stone.",
        num_timesteps=3, 
        random_seed=394856783745,
        image_size=(384,384)
    )

    image_pil.show()

    time_end = time.perf_counter()
    print(f"Elapsed time: {time_end - time_start:.3f} seconds")
