HARDWARE REQUIREMENTS:

  - An NVidia GPU with probably 4GB of VRAM.
  - About 5GB of harddrive space.
  - An internet connection.

SOFTWARE REQUIREMENTS: 

  - Python 3.11.x
  - possibly CUDA drivers.  Just get them to be safe.
  - CUDA aware torch libraries (version 12.1)
  - Not sure, but maybe you'll need to create a HuggingFace account and then get permission to access the Stable Diffusion repository (free).

Currently expects all pip installs to be already run, my apologies.
Also expects a folder at the same level as the parent folder called "AI Models" to exist.

On the first go, will download about 5GB of files.

Has an extremely low VRAM and RAM footprint, should run on almost any hardware with a CUDA compliant GPU.
