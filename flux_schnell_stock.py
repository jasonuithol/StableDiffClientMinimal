# file: flux_schnell_stock.py
import os, torch, time
from diffusers import FluxSchnellPipeline, FlowMatchEulerDiscreteScheduler

from constants import get_model_config

# -------------------------------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------------------------------
MODEL_KEY = "FLUX_SCHNELL"
MODEL_PATH = get_model_config(MODEL_KEY)["MODEL_PATH"]

# -------------------------------------------------------------------------
# CUDA allocator knobs (Windows-safe)
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# PIPELINE INIT
# -------------------------------------------------------------------------
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = FluxSchnellPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)

# Use recommended scheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

# -------------------------------------------------------------------------
# MEMORY-SAFE SETTINGS
# -------------------------------------------------------------------------
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# -------------------------------------------------------------------------
# OPTIONAL: VRAM logging helper
# -------------------------------------------------------------------------
def log_vram(stage):
    if torch.cuda.is_available():
        print(f"[VRAM] {stage}: {torch.cuda.memory_allocated()/1024**2:.2f} MB "
              f"(peak {torch.cuda.max_memory_allocated()/1024**2:.2f} MB)")

# -------------------------------------------------------------------------
# IMAGE GENERATION
# -------------------------------------------------------------------------
def generate_image(prompt, num_timesteps=4, random_seed=42, image_size=(512, 512)):
    width, height = image_size
    assert width % 8 == 0 and height % 8 == 0, "Image dimensions must be divisible by 8"

    g = torch.Generator(device="cpu").manual_seed(random_seed)

    log_vram("before inference")
    image_pil = pipe(
        prompt=prompt,
        num_inference_steps=num_timesteps,
        generator=g,
        width=width,
        height=height,
    ).images[0]
    log_vram("after inference")

    return image_pil

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    time_start = time.perf_counter()

    image_pil = generate_image(
        prompt="a glowing jellyfish in deep ocean",
        num_timesteps=4,
        random_seed=42,
        image_size=(512, 512)
    )

    image_pil.show()

    time_end = time.perf_counter()
    print(f"Elapsed time: {time_end - time_start:.3f} seconds")