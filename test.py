import numpy as np

import numbotics.robot as rob
import numbotics.spatial as spt

import torch

import time


S = 0.0
E = 0.0

def start_time():
    global S
    S = time.perf_counter_ns()

def end_time(label):
    global E
    E = time.perf_counter_ns()
    print(f'{label} time:', (E-S)*1E-9)




arm = rob.Robot('../../research/smm_prediction/learning/7R/7R_new.rob')

pos_samples = 250
res = arm.length/float(pos_samples)

IKS = 100
CORRECT = 100
ITERS = 50

O_RAD = 0.149


loc = spt.trans_mat(pos=np.array([[0.2,1.15,4.0]]).T)


locs = np.resize(loc,(IKS,4,4))

s, qs = arm.batch_ik(locs, ik_iters=150, thresh=1e-4)
size = qs.shape[0]

n_I = np.zeros((size,arm.n,arm.n))
n_I[:,[i for i in range(arm.n)],[i for i in range(arm.n)]] = 1.0

# min_sings_o = np.ones((size,))*np.inf
# min_sings_p = np.ones((size,))*np.inf

min_sings = np.ones((size,))*np.inf

Js = arm.batch_jac(qs)
sig = np.linalg.svd(Js, compute_uv=False)
for i in range(ITERS):

    d_sig = arm.batch_d_sigma(Js,arm.batch_hess(qs))
    grad_sigma_min = d_sig[np.arange(size),np.argmin(sig,axis=1),:,np.newaxis]

    Jp = np.linalg.pinv(Js)
    dq = grad_sigma_min.squeeze(2)

    qs -= dq*0.25

    fks = arm.batch_fk(qs)

    p_valid = np.zeros((size,), dtype=int)
    o_valid = np.zeros((size,), dtype=int)
    print('starting err correction')
    for j in range(CORRECT):
        p_diffs = np.linalg.norm(locs[:,0:3,3]-fks[:,0:3,3],ord=2,axis=1)
        # o_diffs = np.linalg.norm(locs[:,0:3,2]-fks[:,0:3,2],ord=2,axis=1)
        o_cross = np.cross(locs[:,0:3,2],fks[:,0:3,2])
        o_vec = o_cross/np.linalg.norm(o_cross,ord=2,axis=1)[:,np.newaxis]
        o_ang = np.arccos((locs[:,np.newaxis,0:3,2]@fks[:,0:3,2,np.newaxis])[:,0,0])
        p_valid = np.where(p_diffs < res/2.0, 1, 0)
        o_valid = np.where(o_ang < O_RAD, 1, 0)

        p_inds = np.where(p_valid)
        o_inds = np.where(o_valid)

        print(f'pos vals: {p_inds[0].shape[0]}, orn vals: {o_inds[0].shape[0]}')
        if ((p_inds[0].shape[0] == size) and (o_inds[0].shape[0] == size)):
            break

        dx = arm.batch_fk_err(locs,qs)
        dx[:,3:6] = -o_cross[:,:,np.newaxis]

        dx[p_inds,0:3] = 0.0
        dx[o_inds,3:6] = 0.0

        Js = arm.batch_jac(qs)

        d_sig = arm.batch_d_sigma(Js,arm.batch_hess(qs))
        grad_sigma_min = d_sig[np.arange(size),np.argmin(sig,axis=1),:,np.newaxis]

        Jp = np.linalg.pinv(Js)
        dq = ((n_I - (Jp@Js))@grad_sigma_min).squeeze(2)
        qs += (((np.linalg.pinv(Js)@dx).squeeze(2) - dq)*0.1)
        fks = arm.batch_fk(qs)

    valid = np.intersect1d(p_inds,o_inds)

    Js = arm.batch_jac(qs)

    # sig_o = np.linalg.svd(Js[valid,3:6,:], compute_uv=False)
    # sig_p = np.linalg.svd(Js[valid,:3,:], compute_uv=False)

    # min_sings_o[valid] = np.minimum(np.amin(sig_o[valid],axis=1),min_sings_o[valid])
    # min_sings_p[valid] = np.minimum(np.amin(sig_p[valid],axis=1),min_sings_p[valid])

    sig = np.linalg.svd(Js, compute_uv=False)

    min_sings[valid] = np.minimum(np.amin(sig[valid],axis=1), min_sings[valid])

print(np.amin(min_sings))
# print(np.amin(min_sings_p))

fks = arm.batch_fk(qs)
print(fks[0])
