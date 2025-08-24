import torch, os, random, time
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

from model_asset_handler import model_asset_handler
from constants import get_model_config

# Do not remove
MODEL_KEY = "FLUX_DEV"
MODEL_PATH = get_model_config(MODEL_KEY)["MODEL_PATH"]
model_asset_handler(MODEL_KEY)

# allocator knobs
if os.name == "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

quant_config = BitsAndBytesConfig(load_in_4bit=True)
text_encoder_4bit = T5EncoderModel.from_pretrained(
    MODEL_PATH,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
transformer_4bit = FluxTransformer2DModel.from_pretrained(
    MODEL_PATH,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = FluxPipeline.from_pretrained(
    MODEL_PATH,
    text_encoder_2=text_encoder_4bit,
    transformer=transformer_4bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

pipeline.text_encoder.to("cuda")

def generate_image(prompt, image_size):

    random_seed = random.randint(0, 2**32 - 1)  # 32-bit unsigned int range

    image_pil = pipeline(
        prompt, 
        guidance_scale=3.5, 
        height=384, 
        width=384, 
        num_inference_steps=20
    ).images[0]

    return image_pil

#
# MAIN 
#
if __name__ == "__main__":

    time_start = time.perf_counter()  # High‑resolution timer

    image_pil = generate_image(
        prompt="You have entered a dark crypt made of crumbling stone.  Distant echoes of collapsing piles of rubble and bones disturb the otherwise deathly silence.  Strange diagrams and ancient runes decorate the odd stone.",
        image_size=(384,384)
    )

    image_pil.show()

    time_end = time.perf_counter()
    print(f"Elapsed time: {time_end - time_start:.3f} seconds")
