# file: flux_dev_stock.py
import os, torch, random
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler

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

MODEL_PATH = r"..\AI Models\black-forest-labs_FLUX.1-dev"  # or your local path

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

g = torch.Generator(device="cpu").manual_seed(42)
image = pipe(
    prompt="a glowing jellyfish in deep ocean",
    num_inference_steps=24,
    generator=g,
    width=768, height=768,
).images[0]

image.save("flux_stock.png")
