import numpy as np
from scipy.interpolate import BSpline



def unit_bspline(control_points: np.ndarray, degree: int = 1):
        control_points = np.asarray(control_points)
        if control_points.ndim != 2:
            raise ValueError("control_points must be a 2D array (B x n)")
        degree = degree
        
        B, _ = control_points.shape
        if degree >= B:
            raise ValueError("Degree must be less than the number of control points")

        knots = np.concatenate((
            np.zeros(degree),
            np.linspace(0, 1, B - degree + 1),
            np.ones(degree)
        ))
        return BSpline(knots, control_points, degree)

