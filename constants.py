from typing import Dict

LOCAL_MODELS_PATH = r"C:\Users\jason\Desktop\AI Models"

STABLE_DIFFUSION = {
    "MODEL_PATH": f"{LOCAL_MODELS_PATH}/stable-diffusion-v1-5",
    "REPOSITORY_ROOT": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"
}

FLUX_DEV = {
    "MODEL_PATH": f"{LOCAL_MODELS_PATH}/black-forest-labs_FLUX.1-dev",
    "REPOSITORY_ROOT": "https://huggingface.co/black-forest-labs/FLUX.1-dev"
}

FLUX_SCHNELL = {
    "MODEL_PATH": f"{LOCAL_MODELS_PATH}/black-forest-labs_FLUX.1-schnell",
    "REPOSITORY_ROOT": "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
}


def get_model_config(key: str) -> Dict[str, str]:
    cfg = globals()[key]
    return {**cfg, "LOCAL_MODELS_PATH": LOCAL_MODELS_PATH}

