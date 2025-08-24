# file: flux_dev_stock.py
import os, torch, time, random
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler

from constants import get_model_config

# Do not remove
MODEL_KEY = "FLUX_DEV"
MODEL_PATH = get_model_config(MODEL_KEY)["MODEL_PATH"]

# allocator knobs
if os.name == "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)

# use the recommended scheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

# offload everything except the hot path; VAE tiling to keep memory sane
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# optional: reduce peak activation memory a bit
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

def generate_image(prompt, image_size):

    num_timesteps = 28
    random_seed = random.randint(0, 2**32 - 1)  # 32-bit unsigned int range

    if image_size is None:
        image_size = (768,768)

    width, height = image_size

    # TODO: Assert dimensions are correct i.e. mod 16 or something == 0

    g = torch.Generator(device="cpu").manual_seed(random_seed)

    image_pil = pipe(
        prompt=prompt,
        num_inference_steps=num_timesteps,
        generator=g,
        width=width, 
        height=height,
    ).images[0]

    return image_pil

#
# MAIN 
#
if __name__ == "__main__":

    time_start = time.perf_counter()  # High‑resolution timer

    image_pil = generate_image(
        prompt="a glowing jellyfish in deep ocean",
        num_timesteps=28, 
        random_seed=42,
        image_size=(512,512)
    )

    image_pil.show()

    time_end = time.perf_counter()
    print(f"Elapsed time: {time_end - time_start:.3f} seconds")