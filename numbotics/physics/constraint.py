import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

from numbotics.physics import pyb



class Constraint(enum.Enum):
    FIXED = pyb.JOINT_FIXED
    REVOLUTE = pyb.JOINT_REVOLUTE
    PRISMATIC = pyb.JOINT_PRISMATIC
    SPHERICAL = pyb.JOINT_SPHERICAL



@dataclass(frozen=True)
class Joint:
    offset: np.ndarray
    axis: np.ndarray
    type: Constraint
    name: Optional[str] = None
    parent_pose: Optional[np.ndarray] = None
    child_pose: Optional[np.ndarray] = None
    damping: float = 0.01
    lower_limit: float = -np.inf
    upper_limit: float = np.inf
    max_velocity: float = np.inf
    max_effort: float = np.inf

    def __post_init__(self):
        if self.type == Constraint.FIXED:
            object.__setattr__(self, 'axis', np.zeros((3,), dtype=np.float64))
            object.__setattr__(self, 'offset', self.offset.astype(np.float64))
        else:
            object.__setattr__(self, 'axis', self.axis.astype(np.float64))
            object.__setattr__(self, 'offset', self.offset.astype(np.float64))

    
    def __hash__(self):
        if self.name is None:
            return id(self)
        return hash(self.name)


    @property
    def dof(self):
        if self.type == Constraint.FIXED:
            return 0
        elif self.type == Constraint.REVOLUTE:
            return 1
        elif self.type == Constraint.PRISMATIC:
            return 1
        elif self.type == Constraint.SPHERICAL:
            return 3
        else:
            raise ValueError(f'Invalid constraint type: {self.type}')
            