import numpy as np


def is_PD(A: np.ndarray):
    eigvals = np.linalg.eigvals(A)
    eigvals[np.isclose(eigvals, 0)] = 0
    return np.all(eigvals > 0)


def is_PSD(A: np.ndarray):
    eigvals = np.linalg.eigvals(A)
    eigvals[np.isclose(eigvals, 0)] = 0
    return np.all(eigvals >= 0)


def is_ND(A: np.ndarray):
    eigvals = np.linalg.eigvals(A)
    eigvals[np.isclose(eigvals, 0)] = 0
    return np.all(eigvals < 0)


def is_NSD(A: np.ndarray):
    eigvals = np.linalg.eigvals(A)
    eigvals[np.isclose(eigvals, 0)] = 0
    return np.all(eigvals <= 0)


def is_symmetric(A: np.ndarray):
    return np.allclose(A, A.T)


def is_SO3(A: np.ndarray):
    if not np.allclose(A.T @ A, np.eye(3)):
        return False

    if not np.allclose(np.linalg.det(A), 1.0):
        return False

    return True


def is_SE3(A: np.ndarray):
    if not is_SO3(A[:3, :3]):
        return False
    if not np.allclose(A[3, :4], np.array([0, 0, 0, 1])):
        return False
    return True
