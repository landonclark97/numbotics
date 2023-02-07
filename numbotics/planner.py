import numbotics.logger as nlog
import numbotics.robot as rob
import numbotics.solver as nsol
import numbotics.spatial as spt
import numbotics.topology as top
# import numbotics.graphics as gfx

import numpy as np


# smm shape: (H, n, S)
def bounding_box(arm, smm_x0, smm_x1):
    assert (smm_x0.shape[1] == arm.n) and (smm_x1.shape[1] == arm.n)

    a_limits = top.smm_bb_intersect(smm_x0, smm_x1)

    # choose random configuration inside bounding box
