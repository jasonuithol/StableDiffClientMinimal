HARDWARE REQUIREMENTS:

  - An NVidia GPU with probably 4GB of VRAM.
  - About 5GB of harddrive space.
  - An internet connection.

SOFTWARE REQUIREMENTS: 

  - Python 3.11.x
  - possibly CUDA drivers.  Just get them to be safe.
  - CUDA aware torch libraries (version 12.1)
  - Not sure, but maybe you'll need to create a HuggingFace account and then get permission to access the Stable Diffusion repository (free).

CONFIGURATION:

  - You'll very obviously see that there's some local paths to set up in both files: they both point to a single folder that model files will be downloaded and stored.
  - the text prompt, image size and a "random" seed are hardcoded into the example top-level method call for benchmarking purposes.
  - A file called nvml.dll will need to be present and it's location is specified in the file_access_logger.py, you'll see it.

NOTES:

Currently expects all pip installs to be already run, my apologies.
Also expects a folder at the same level as the parent folder called "AI Models" to exist.

On the first go, will download about 5GB of files.

Has an extremely low VRAM and RAM footprint, should run on almost any hardware with a CUDA compliant GPU.
