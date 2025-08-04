import numpy as np

from numbotics.utils import parse_shape_kwargs, Shape



class VisualShape:
    def __init__(self, shape: Shape, offset: np.ndarray | None = None, color: np.ndarray | None = None, **kwargs):
        if not isinstance(shape, Shape):
            raise ValueError(f"Invalid shape type: {shape}")
        self.shape = shape
        self.offset = offset if offset is not None else np.eye(4)
        self.color = color if color is not None else np.array([0.7, 0.7, 0.7, 1.0])
        _, shape_info = parse_shape_kwargs(kwargs)
        self.visual_shape = self.shape.create_visual_shape(**shape_info)