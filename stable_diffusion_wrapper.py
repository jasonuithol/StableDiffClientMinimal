import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import time, torch, pynvml, ctypes, random

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from PIL import Image
from constants import get_model_config
from model_asset_handler import model_asset_handler

# Do not remove
MODEL_KEY = "STABLE_DIFFUSION"
MODEL_PATH = get_model_config(MODEL_KEY)["MODEL_PATH"]
model_asset_handler(MODEL_KEY)


def get_vram_info():
    # Force-load the correct NVML DLL
    nvml_path = "C:\\Windows\\System32\\nvml.dll"
    nvml_lib = ctypes.CDLL(nvml_path)

    # Patch pynvml to use the already-loaded DLL
    pynvml.nvmlLib = nvml_lib

    # Now initialize NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total / 1024**2  # MB
    used = meminfo.used / 1024**2
    free = meminfo.free / 1024**2
    return {"total": total, "used": used, "free": free}

@torch.inference_mode()  # safer, leaner than manual no_grad blocks everywhere
def generate_image(prompt, image_size):

    num_timesteps = 28 
    random_seed = random.randint(0, 2**32 - 1)  # 32-bit unsigned int range

    DTYPE = torch.float16
    SAFE_MARGIN_MB = 70  # adjust based on your tolerance


    width, height = image_size
    assert width % 8 == 0 and height % 8 == 0, "Width and height must be divisible by 8"

    # Step 0: Check available VRAM
    vram = get_vram_info()
    print(f"VRAM info: {vram}")

    # Estimate VRAM usage for target size
    est_usage = (width * height * 3 * 2) / 1024**2 * 1.5  # rough multiplier

    if est_usage > vram["free"] - 1000:
        print(f"⚠️ Estimated usage ({est_usage:.1f} MB) exceeds safe margin. Aborting.")
        exit()

    # 🧩 Step 1: Text Encoding
    r'''
    stable-diffusion-v1-5\tokenizer\merges.txt
    stable-diffusion-v1-5\tokenizer\tokenizer_config.json
    '''
    tokenizer = CLIPTokenizer.from_pretrained(
        f"{MODEL_PATH}", 
        subfolder="tokenizer"
    )
    print("Created tokenizer")

    r'''
    stable-diffusion-v1-5\text_encoder\config.json
    '''
    text_encoder = CLIPTextModel.from_pretrained(
        f"{MODEL_PATH}", 
        subfolder="text_encoder"
    ).to(device="cuda", dtype=DTYPE)
    print("Created text_encoder")

    # Tokenize prompt
    inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
    del tokenizer

    # Encode to get text embeddings
    text_embeddings = text_encoder(inputs.input_ids)[0]  # shape: [batch, seq_len, hidden_dim]
    del text_encoder
    del inputs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("Freed memory after embedding prompt.")

    text_embeddings = text_embeddings.to(dtype=DTYPE) 
    print("Created text_embeddings")

    # 🧊 Step 2: Initial Latent
    torch.manual_seed(random_seed)
    latent_h = height // 8
    latent_w = width // 8
    latent = torch.randn([1, 4, latent_h, latent_w], dtype=DTYPE, device="cuda")

    # 🔁 Step 3: Denoising Loop with Cross-Attention
    r'''
    stable-diffusion-v1-5\unet\config.json
    stable-diffusion-v1-5\unet\diffusion_pytorch_model.safetensors
    '''
    unet = UNet2DConditionModel.from_pretrained(
        f"{MODEL_PATH}/unet",  # local path
        torch_dtype=DTYPE
    ).to("cuda")

    unet.to(memory_format=torch.channels_last)
    try:
        from diffusers.models.attention_processor import SlicedAttnProcessor
        unet.set_attn_processor(SlicedAttnProcessor())
        print("Created unet (with SlicedAttnProcessor wrapper)")
    except Exception:
        print("Created unet (unwrapped)")


    # Scheduler setup
    r'''
    stable-diffusion-v1-5\scheduler\scheduler_config.json
    '''
    scheduler = DDIMScheduler.from_pretrained(
        f"{MODEL_PATH}/scheduler",  # local path
        subfolder=None
    )
    scheduler.set_timesteps(num_timesteps)
    timesteps = scheduler.timesteps #.to(device="cuda")

    print("Created scheduler")
    
    # Denoising loop
    min_free_vram_mb = 16 * 1024 # some arbitrarily big number
    for i, t in enumerate(timesteps):
        print(f"Entering denoising step {i}")

        vram = get_vram_info()
        min_free_vram_mb = min(min_free_vram_mb, vram['free'])

        if vram["free"] < SAFE_MARGIN_MB:
            print(f"❌ VRAM low ({vram['free']:.1f} MB free). Aborting at step {i}.")
            break

        t_tensor = t.expand(1).to(device="cuda", dtype=torch.float32)

        noise_pred = unet(latent, t_tensor, text_embeddings).sample
        del t_tensor

        latent = scheduler.step(noise_pred, t, latent).prev_sample
        del noise_pred

    print(f"Denoising completed. {min_free_vram_mb} MB of VRAM was left unused by loop.")

    del unet
    del scheduler
    del timesteps
    del text_embeddings
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Freed memory after denoising.")

    # 🎨 Step 4: Decode Final Image
    r'''
    stable-diffusion-v1-5\vae\config.json
    stable-diffusion-v1-5\vae\diffusion_pytorch_model.safetensors    
    '''
    vae = AutoencoderKL.from_pretrained(
        f"{MODEL_PATH}",
        subfolder="vae"
    )
    vae = vae.to(dtype=torch.float32, device="cpu")
    vae.enable_tiling()
    vae.enable_slicing()

    print("Created vae")

    latent = latent.detach().to(device="cpu", dtype=torch.float32)
    latent = latent / 0.18215 # Stable-Diffusion 1.5 scaling
    print(f"Detached and moved latent image tensor to CPU/float32")

    image_tensor = vae.decode(latent).sample # [1, 3, 512, 512]
    del vae
    del latent
    print("Decoded latent image tensor into decoded image tensor using vae.")

    #image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2  # scale to [0, 1]
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()  # [1, H, W, C]

    image_numpy = (image_tensor[0] * 255).round().astype("uint8")  # [H, W, C]
    del image_tensor

    image_pil = Image.fromarray(image_numpy)
    del image_numpy

    print("Decoded final image")
    return image_pil

#
# MAIN 
#
if __name__ == "__main__":

    time_start = time.perf_counter()  # High‑resolution timer

    image_pil = generate_image(
        prompt="a glowing jellyfish in deep ocean",
        image_size=(512,512)
    )

    image_pil.show()

    time_end = time.perf_counter()
    print(f"Elapsed time: {time_end - time_start:.3f} seconds")


