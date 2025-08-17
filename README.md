WHY ARE YOU HERE ?

- Minimal, GPU‑aware Stable Diffusion client — distilled to the essentials for lean, reproducible image generation with explicit VRAM control.
- CUDA 12.1 + PyTorch integration — pinned, tested, and documented so it runs where it’s supposed to, without dependency roulette.
- Transparent architecture — every dependency justified, every path configurable, every large asset cached and tracked.
- Only the bare essentials downloaded - automatically, and once.
- 75% Vibe coded from only the purest vibes.

HARDWARE REQUIREMENTS:

  - An NVidia GPU with probably 4GB of VRAM.
  - About 5GB of harddrive space.
  - An internet connection.

SOFTWARE REQUIREMENTS: 

  - Python 3.11.x
  - possibly CUDA drivers.  Just get them to be safe.
  - A full installation of Nvidia driver software that includes nvml.dll
  - CUDA aware torch libraries (version 12.1)
  - Not sure, but maybe you'll need to create a HuggingFace account and then get permission to access the Stable Diffusion 1.5 repository (free).

INSTALLATION:

  - Ensure software requirements are met.
  - Either pull the repo, or manually download these files.
  - Run this command: pip install -r requirements.txt

CONFIGURATION:

  - You'll very obviously see that there's some local paths to set up in both files: they both point to a single folder that model files will be downloaded and stored.
  - the text prompt, image size and a "random" seed are hardcoded into the example top-level method call for benchmarking purposes.
  - A file called nvml.dll will need to be present and it's location is specified in the file_access_logger.py, you'll see it.

EXECUTION:

Running "python image.py" is sufficient.

NOTES:

Expects a folder at the same level as the parent folder called "AI Models" to exist.  You can edit this path - check the images.py and file_access_logger.py files for a variable called LOCAL_MODELS_PATH and change to suit your needs if required.

On the first go, will download about 5GB of files.

Has an extremely low VRAM and RAM footprint, should run on almost any hardware with a CUDA compliant GPU.
