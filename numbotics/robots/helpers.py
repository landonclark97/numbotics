import numpy as np
from numba import njit, float64, int64, prange, boolean
from numba.types import Optional

from numbotics.physics import Constraint



@njit(float64[:,:,:](float64[:,:,:], float64[:,:,:]), cache=True, fastmath=True)
def nb_batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    return np.sum(
        np.multiply(A[:, :, :, np.newaxis], B[:, np.newaxis, :, :]),
        axis=2
    )


@njit([float64[:,:,:](float64[:,:]), float64[:,:](float64[:])], cache=True, fastmath=True)
def nb_skew(v: np.ndarray) -> np.ndarray:
    assert v.shape[-1] == 3
    shape = (*v.shape[:-1], 3, 3)
    skew = np.zeros(shape)
    skew[..., 0, 1] = -v[..., 2]
    skew[..., 0, 2] =  v[..., 1]
    skew[..., 1, 0] =  v[..., 2]
    skew[..., 1, 2] = -v[..., 0]
    skew[..., 2, 0] = -v[..., 1]
    skew[..., 2, 1] =  v[..., 0]
    return skew


@njit(float64[:,:,:](float64[:,:], float64[:], float64[:], int64), cache=True, fastmath=True)
def nb_joint_transform(offset: np.ndarray, axis: np.ndarray, q: np.ndarray, joint_type: int) -> np.ndarray:
    B = q.shape[0]
    
    if joint_type == Constraint.FIXED.value:
        offset = offset.repeat(B).reshape((4, 4, B))
        offset = np.swapaxes(offset, 0, -1)
        offset = np.swapaxes(offset, -1, -2)
        return offset
    
    elif joint_type == Constraint.REVOLUTE.value:
        R = np.ascontiguousarray(offset[:3, :3])
        K = np.ascontiguousarray(np.outer(axis, axis))
        K_eye = np.ascontiguousarray(K - np.eye(3))
        K_skew = np.ascontiguousarray(nb_skew(axis))
        sin_q = np.sin(q)
        cos_q = np.cos(q)
        rotated_offsets = np.zeros((B, 4, 4))
        rotated_offsets[:,3,3] = 1.0
        for i in prange(B):
            rotated_offsets[i, :3, :3] = R @ (K - (cos_q[i] * K_eye) + (sin_q[i] * K_skew)) 
            rotated_offsets[i, :3, 3] = offset[:3, 3]
        return rotated_offsets
    
    elif joint_type == Constraint.PRISMATIC.value:
        R = np.ascontiguousarray(offset[:3, :3])
        dist_along_axis = (R @ np.ascontiguousarray(axis)).repeat(B).reshape((3, B))
        dist_along_axis = np.swapaxes(dist_along_axis, 0, -1)
        dist_along_axis *= q
        translated_offsets = np.zeros((B, 4, 4))
        translated_offsets[:,3,3] = 1.0
        translated_offsets[:, :3, 3] = offset[:3, 3] + dist_along_axis
        return translated_offsets
    
    elif joint_type == Constraint.SPHERICAL.value:
        _q = np.ascontiguousarray(q).reshape(-1, 3)
        mag = np.sqrt(np.sum(_q * _q, axis=1))
        dir = _q / mag.reshape((B, 1))
        B = _q.shape[0]
        R = np.ascontiguousarray(offset[:3, :3])
        K = np.ascontiguousarray(nb_skew(dir))
        K2 = np.zeros((B, 3, 3))
        for i in prange(B):
            K2[i] = np.ascontiguousarray(K[i] @ K[i])
        sin_q = np.sin(mag).reshape((B, 1, 1))
        cos_q = np.cos(mag).reshape((B, 1, 1))
        local_R = np.eye(3) + K * sin_q + K2 * (1.0 - cos_q)
        rotated_offsets = np.zeros((B, 4, 4))
        rotated_offsets[:,3,3] = 1.0
        for i in prange(B):
            rotated_offsets[i, :3, :3] = np.dot(R, np.ascontiguousarray(local_R[i]))
            rotated_offsets[i, :3, 3] = offset[:3, 3]
        return rotated_offsets
    
    else:
        return np.zeros((B, 4, 4))


@njit(float64[:,:,:](float64[:,:,:], float64[:,:,:], float64[:,:], int64[:], int64[:], float64[:,:]), cache=True, fastmath=True)
def nb_compute_transformation(T: np.ndarray, offsets: np.ndarray, axes: np.ndarray, joint_types: np.ndarray, joint_idxs: np.ndarray, q: np.ndarray) -> np.ndarray:
    B = q.shape[0]
    J = offsets.shape[0]
    T = np.ascontiguousarray(T)
    for i in range(J):
        offset = offsets[i]
        axis = axes[i]
        joint_type = joint_types[i]
        joint_idx = joint_idxs[i]
        if joint_type == Constraint.SPHERICAL.value:
            transforms = np.ascontiguousarray(nb_joint_transform(offset, axis, q[:, joint_idx].flatten(), joint_type))
            for i in prange(B):
                T[i] = T[i] @ transforms[i]
        elif joint_type != Constraint.FIXED.value:
            transforms = np.ascontiguousarray(nb_joint_transform(offset, axis, q[:, joint_idx], joint_type))
            for i in prange(B):
                T[i] = T[i] @ transforms[i]
        else:
            transforms = np.ascontiguousarray(nb_joint_transform(offset, axis, np.zeros(q.shape[0]), joint_type))
            for i in prange(B):
                T[i] = T[i] @ transforms[i]
    return T



@njit(float64[:,:,:](float64[:,:,:], float64[:,:,:,:], float64[:,:], float64[:,:,:], float64[:,:], int64[:], int64[:], float64[:,:], Optional(float64[:,:,:]), Optional(float64[:,:,:]), boolean), cache=True, fastmath=True)
def nb_compute_jacobian(
    T: np.ndarray, 
    T_mats: np.ndarray,
    com_offset: np.ndarray,
    offsets: np.ndarray,
    axes: np.ndarray,
    joint_types: np.ndarray,
    joint_idxs: np.ndarray,
    q: np.ndarray,
    local_pose: np.ndarray | None = None,
    global_pose: np.ndarray | None = None,
    use_com: bool = False,
) -> np.ndarray:
    B = q.shape[0]
    I = axes.shape[0]
    J = np.ascontiguousarray(np.zeros((B, 6, q.shape[1])))

    T_mats = np.ascontiguousarray(T_mats)
    T = np.ascontiguousarray(T)

    for i in range(I):
        offset = np.ascontiguousarray(offsets[i])
        axis = np.ascontiguousarray(axes[i])
        joint_type = joint_types[i]
        joint_idx = joint_idxs[i]
        
        if joint_type == Constraint.SPHERICAL.value:
            raise NotImplementedError("Jacobian for spherical joints not implemented")
        elif joint_type != Constraint.FIXED.value:
            transforms = np.ascontiguousarray(nb_joint_transform(offset, axis, q[:,joint_idx], joint_type))
            for j in prange(B):
                T[j, :, :] = np.ascontiguousarray(T[j, :, :]) @ transforms[j]
            T_mats[:, i, :, :] = T
        else:
            transforms = np.ascontiguousarray(nb_joint_transform(offset, axis, np.zeros((B,)), joint_type))
            for j in prange(B):
                T[j, :, :] = np.ascontiguousarray(T[j, :, :]) @ transforms[j]
            T_mats[:, i, :, :] = T

    for i in prange(B):
        T_mats[i, -1, :, :] = np.ascontiguousarray(T[i, :, :]) @ np.ascontiguousarray(offsets[-1])

    if use_com:
        for i in prange(B):
            T_mats[i, -1, :, :] = np.ascontiguousarray(T_mats[i, -1, :, :]) @ np.ascontiguousarray(com_offset)

    if local_pose is not None:
        for i in prange(B):
            T_mats[i, -1, :, :] = np.ascontiguousarray(T_mats[i, -1, :, :]) @ np.ascontiguousarray(local_pose[i, :, :])

    if global_pose is not None:
        for i in prange(B):
            T_mats[i, -1, :, :] = np.ascontiguousarray(global_pose[i, :, :])

    for i in prange(I):
        axis = np.ascontiguousarray(axes[i])
        joint_type = joint_types[i]
        joint_idx = joint_idxs[i]
        if joint_type == Constraint.FIXED.value:
            continue

        for j in prange(B):
            w_axis = np.ascontiguousarray(T_mats[j, i, :3, :3]) @ axis
            if joint_type == Constraint.REVOLUTE.value:
                J[j, :3, joint_idx] = np.cross(w_axis, T_mats[j, -1, :3, 3] - T_mats[j, i, :3, 3])
                J[j, 3:, joint_idx] = w_axis
            elif joint_type == Constraint.PRISMATIC.value:
                J[j, :3, joint_idx] = w_axis
    
    return J

