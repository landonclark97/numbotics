import subprocess

import numbotics.utils.logger as logger
from numbotics.utils import pipes

import pathlib

VERBOSE = True
USE_TORCH = True
USE_GFX = True

# Check for torch availability
try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

# Check for matplotlib availability
try:
    import matplotlib

    MATPLOT_AVAIL = True
except ImportError:
    MATPLOT_AVAIL = False

# Set torch device if available
if TORCH_AVAIL and USE_TORCH:
    if torch.cuda.is_available():
        TORCH_DEV = torch.device("cuda")
    elif torch.backends.mps.is_available():
        TORCH_DEV = torch.device("mps")
    else:
        TORCH_DEV = torch.device("cpu")
else:
    TORCH_DEV = None

try:
    with pipes():
        subprocess.call(["ffmpeg", "-version"])
    FFMPEG_AVAIL = True
except FileNotFoundError:
    FFMPEG_AVAIL = False
    logger.warning("ffmpeg is not installed, video recording will raise an error...")
