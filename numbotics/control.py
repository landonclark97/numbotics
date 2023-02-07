import numbotics.robot as rob
import numbotics.logger as nlog
import numbotics.solver as nsol
import numbotics.spatial as spt
import numbotics.config as conf

import numpy as np
import numpy.matlib
import scipy
import scipy.linalg

import matplotlib.pyplot as plt


def grad_sigma_sum(d_sig, sig):
    return np.sum(d_sig,axis=0)[...,np.newaxis]


def grad_sigma_prod(d_sig, sig):
    ds = np.matlib.repmat(sig,sig.shape[0],1)
    np.fill_diagonal(ds,1.0)
    ds = np.prod(ds,axis=1)[...,np.newaxis]
    return d_sig.T@ds


def grad_sigma_min(d_sig, sig):
    return d_sig[np.argmin(sig),:,np.newaxis]


def rr_tracking(arm, x_d, dt=0.05, max_vel=0.5, dls_lambda=0.1, dist_thresh=1e-4, sing_avoid=None, disp=False):
    if sing_avoid is not None:
        assert sing_avoid in [grad_sigma_min, grad_sigma_prod, grad_sigma_sum]
    if disp:
        assert arm.use_gfx
        import numbotics.graphics
        gfx = numbotics.graphics

    sing_comp = np.zeros((arm.n,1))

    dx = arm.fk_err(x_d, mask=True)
    while np.linalg.norm(dx) > dist_thresh:
        J = arm.jac

        dq = (J.T@np.linalg.inv((J@J.T)+(dls_lambda*np.eye(arm.m))))@dx
        vel = np.linalg.norm(dq)

        if sing_avoid is not None:
            d_sig = arm.d_sigma(J,arm.hess)
            sig = np.linalg.svd(J,compute_uv=False)
            sing_comp = (np.eye(arm.n)-(np.linalg.pinv(J)@J))@sing_avoid(d_sig,sig)
            dq += sing_comp
            vel = np.linalg.norm(dq)

        dq /= vel
        if vel > max_vel:
            vel = max_vel
        arm.q += (dq*vel*dt)

        dx = arm.fk_err(x_d, mask=True)
        if vel < 1e-4:
            break
        print(vel)

        if disp:
            arm.update_gfx()
            gfx.gfx_rate(1.0/dt)

    if np.linalg.norm(dx) < dist_thresh:
        return True
    return False


def rr_waypoints(arm, waypoints, dt=0.05, max_vel=0.5, dls_lambda=0.1, dist_thresh=1e-4, sing_avoid=None, disp=False):
    for i, wp in enumerate(waypoints):
        nlog.info(f'tracking workspace location: \n{wp}')
        s = rr_tracking(arm, wp, dt=dt, max_vel=max_vel, dls_lambda=dls_lambda, dist_thresh=dist_thresh, sing_avoid=sing_avoid, disp=disp)
        if not s:
            return False, i
    return True, -1


if __name__ == '__main__':
    r = rob.Robot('../test.rob')
    # r.q = np.array([-np.pi/2.0,1e-6,1e-6])
    wp = []
    wp.append(spt.trans_mat(orn=spt.eul_ZYZ(1.1,0.2,3.3),pos=np.array([[-2.1,0.5,-0.1]]).T))
    wp.append(spt.trans_mat(orn=spt.eul_ZYZ(-1.1,1.2,-3.3),pos=np.array([[-0.2,0.3,0.1]]).T))
    wp.append(spt.trans_mat(orn=spt.eul_ZYZ(1.5,0.8,0.3),pos=np.array([[0.1,0.1,0.9]]).T))
    wp.append(spt.trans_mat(orn=spt.eul_ZYZ(1.1,0.7,-1.1),pos=np.array([[-0.4,0.5,0.0]]).T))
    wp.append(spt.trans_mat(orn=spt.eul_ZYZ(0.0,0.2,2.2),pos=np.array([[-0.2,0.2,0.7]]).T))
    # rr_tracking(r, spt.trans_mat(pos=np.array([[-0.2,0.5,0.2]]).T), disp=True, dist_thresh=1e-2)
    rr_waypoints(r, wp, disp=True, dist_thresh=5e-2, max_vel=0.5, sing_avoid=None)
    del r
