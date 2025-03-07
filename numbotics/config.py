import pkg_resources
import sys

VERBOSE = True

USE_TORCH = True

TORCH_AVAIL = 'torch' in {pkg.key for pkg in pkg_resources.working_set}
MATPLOT_AVAIL = 'matplotlib' in {pkg.key for pkg in pkg_resources.working_set}

if TORCH_AVAIL and USE_TORCH:
    import torch
    if torch.cuda.is_available():
        TORCH_DEV = torch.device('cuda')
    elif torch.backends.mps.is_available():
        TORCH_DEV = torch.device('mps')
    else:
        TORCH_DEV = torch.device('cpu')
