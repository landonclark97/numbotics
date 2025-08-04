import numpy as np
from scipy.optimize import minimize

from numbotics.math.optimization import SO3_constraint



R = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


def obj(X: np.ndarray):
    X = X.reshape(3, 3)
    return np.sum(np.abs((X.T @ R) - np.eye(3)))


res = minimize(
    obj,
    np.random.rand(9),
    method='SLSQP',
    constraints=SO3_constraint(np.arange(9)),
    options={'disp': True},
)
print(res.x.reshape(3, 3))
