import pkg_resources
import sys

VERBOSE = True

USE_TORCH = True

TORCH_AVAIL = 'torch' in {pkg.key for pkg in pkg_resources.working_set}
MATPLOT_AVAIL = 'matplotlib' in {pkg.key for pkg in pkg_resources.working_set}

if TORCH_AVAIL and USE_TORCH:
    import torch
    TORCH_DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
