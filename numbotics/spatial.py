import numpy as np

import numbotics.config as conf

if conf.TORCH_AVAIL and conf.USE_TORCH:
    import torch

def rotx(a):
    if isinstance(a, np.ndarray) or isinstance(a, float):
        a = np.atleast_1d(a)
        assert len(a.shape) <= 2
        if len(a.shape) == 2:
            assert a.shape[1] == 1
        rx = np.reshape(np.tile(np.eye(3),a.shape[0]).T, (a.shape[0],3,3))
        rx[:,1,1] = np.cos(a[:])
        rx[:,1,2] = -np.sin(a[:])
        rx[:,2,1] = np.sin(a[:])
        rx[:,2,2] = np.cos(a[:])
        if rx.shape[0] == 1:
            rx = rx.squeeze(0)
        return rx
    else:
        assert conf.TORCH_AVAIL and conf.USE_TORCH
        assert isinstance(a, torch.Tensor)
        device = conf.TORCH_DEV
        a = torch.atleast_1d(a).to(device)
        assert len(a.shape) <= 2
        if len(a.shape) == 2:
            assert a.shape[1] == 1
        rx = torch.reshape(torch.tile(torch.eye(3),(a.shape[0],)).T, (a.shape[0],3,3))
        rx[:,1,1] = torch.cos(a[:])
        rx[:,1,2] = -torch.sin(a[:])
        rx[:,2,1] = torch.sin(a[:])
        rx[:,2,2] = torch.cos(a[:])
        if rx.shape[0] == 1:
            rx = rx.squeeze(0)
        return rx

def roty(a):
    if isinstance(a, np.ndarray) or isinstance(a, float):
        a = np.atleast_1d(a)
        assert len(a.shape) <= 2
        if len(a.shape) == 2:
            assert a.shape[1] == 1
        ry = np.reshape(np.tile(np.eye(3),a.shape[0]).T, (a.shape[0],3,3))
        ry[:,0,0] = np.cos(a[:])
        ry[:,0,2] = np.sin(a[:])
        ry[:,2,0] = -np.sin(a[:])
        ry[:,2,2] = np.cos(a[:])
        if ry.shape[0] == 1:
            ry = ry.squeeze(0)
        return ry
    else:
        assert conf.TORCH_AVAIL and conf.USE_TORCH
        assert isinstance(a, torch.Tensor)
        device = conf.TORCH_DEV
        a = torch.atleast_1d(a).to(device)
        assert len(a.shape) <= 2
        if len(a.shape) == 2:
            assert a.shape[1] == 1
        ry = torch.reshape(torch.tile(torch.eye(3),(a.shape[0],)).T, (a.shape[0],3,3))
        ry[:,0,0] = torch.cos(a[:])
        ry[:,0,2] = torch.sin(a[:])
        ry[:,2,0] = -torch.sin(a[:])
        ry[:,2,2] = torch.cos(a[:])
        if ry.shape[0] == 1:
            ry = ry.squeeze(0)
        return ry

def rotz(a):
    if isinstance(a, np.ndarray) or isinstance(a, float):
        a = np.atleast_1d(a)
        assert len(a.shape) <= 2
        if len(a.shape) == 2:
            assert a.shape[1] == 1
        rz = np.reshape(np.tile(np.eye(3),a.shape[0]).T, (a.shape[0],3,3))
        rz[:,0,0] = np.cos(a[:])
        rz[:,0,1] = -np.sin(a[:])
        rz[:,1,0] = np.sin(a[:])
        rz[:,1,1] = np.cos(a[:])
        if rz.shape[0] == 1:
            rz = rz.squeeze(0)
        return rz
    else:
        assert conf.TORCH_AVAIL and conf.USE_TORCH
        assert isinstance(a, torch.Tensor)
        device = conf.TORCH_DEV
        a = torch.atleast_1d(a).to(device)
        assert len(a.shape) <= 2
        if len(a.shape) == 2:
            assert a.shape[1] == 1
        rz = torch.reshape(torch.tile(torch.eye(3),(a.shape[0],)).T, (a.shape[0],3,3))
        rz[:,0,0] = torch.cos(a[:])
        rz[:,0,1] = -torch.sin(a[:])
        rz[:,1,0] = torch.sin(a[:])
        rz[:,1,1] = torch.cos(a[:])
        if rz.shape[0] == 1:
            rz = rz.squeeze(0)
        return rz


def eul_ZYZ(psi, tht, phi):
    assert type(psi) is type(tht)
    assert type(psi) is type(phi)
    return rotz(psi) @ roty(tht) @ rotz(phi)


def eul_zyz(psi, tht, phi):
    assert type(psi) is type(tht)
    assert type(psi) is type(phi)
    return rotz(phi) @ roty(tht) @ rotz(psi)


# arbitrary = False will return axis of rotation of difference angle
def rot_diff(A, B, arbitrary=True):
    assert type(A) is type(B)
    if isinstance(A, np.ndarray):
        R = A @ np.swapaxes(B, len(B.shape)-2, len(B.shape)-1)
        T = (np.trace(R, axis1=len(R.shape)-2, axis2=len(R.shape)-1)-1.0)/2.0
        T = np.clip(T, -1.0, 1.0)
        ang = np.arccos(T)
        if arbitrary:
            return ang
        return ang, np.real(np.linalg.eig(R)[1][2,:])
    else:
        assert conf.TORCH_AVAIL and conf.USE_TORCH
        assert isinstance(A, torch.Tensor)
        device = conf.TORCH_DEV
        R = A.to(device) @ torch.swapaxes(B.to(device), len(B.shape)-2, len(B.shape)-1)
        diag_ind = [0,1,2]
        T = torch.sum(R[...,diag_ind,diag_ind],dim=len(R.shape)-2).double()
        T = torch.clamp((T-1.0)/2.0, min=-1.0, max=1.0)
        ang = torch.acos(T)
        if arbitrary:
            return ang
        return ang, torch.real(torch.linalg.eig(R)[1][2,:])


def trans_mat(orn=np.eye(3), pos=np.zeros((3,1))):
    return np.block([[orn, pos], [np.zeros((1,3)), np.ones((1,1))]])


def skew(v):
    assert (v.shape == (3,1))
    return np.array([[0.0, -v[2,0], v[1,0]],
                     [v[2,0], 0.0, -v[0,0]],
                     [-v[1,0], v[0,0], 0.0]])


def skew_mat(v):
    assert (v.shape == (3,1))
    return np.block([[np.eye(3), -skew(v)], [np.zeros((3,3)), np.eye(3)]])
