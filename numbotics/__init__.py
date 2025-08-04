import numbotics.robot
import numbotics.graphics
import numbotics.math.geometry.polytope
import numbotics.utils.logger
import numbotics.solver
import numbotics.math.spatial
import numbotics.planner
import numbotics.control
import numbotics.topology
import numbotics.config

if not numbotics.config.TORCH_AVAIL:
    import sys

    if numbotics.config.VERBOSE:
        numbotics.logger.warning("PyTorch not installed - learning module unavailable")
    sys.modules["numbotics.learning"] = None
else:
    import numbotics.learning
