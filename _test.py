import numpy as np

import numbotics.robot as rob
import numbotics.spatial as spt

import torch

import time

import pickle

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


S = 0.0
E = 0.0

def start_time():
    global S
    S = time.perf_counter_ns()

def end_time(label):
    global E
    E = time.perf_counter_ns()
    print(f'{label} time:', (E-S)*1E-9)





def print_coord(loc):
    # print(loc)
    n = []
    this_o = (loc)%v_shape
    this_z = ((loc-this_o)//v_shape)%(pos_samples+pos_samples+1)
    this_x = (loc-this_o-(this_z*v_shape))//((pos_samples+pos_samples+1)*v_shape)
    # print(f'x: {this_x}, z: {this_z}, orn: {this_o}')
    this_val = (this_x*(pos_samples+pos_samples+1)*v_shape) + (this_z*v_shape) + this_o
    if loc != this_val:
        print('index err')
        quit()

with open('../../tmp/g30_data.pkl', 'rb') as f:
    verts = pickle.load(f)



arm = rob.Robot('../../research/smm_prediction/learning/7R/7R_new.rob')

verts = verts['v']
v_shape = verts.shape[0]
# verts = np.resize(verts,(SAMPLES,verts.shape[0],verts.shape[1]))
# verts = verts[np.newaxis,...]
verts = torch.from_numpy(verts[np.newaxis,...]).to(device)

SAMPLES = 250000

ITERS = 100


# all_inds = np.empty((0,), dtype=int)
all_inds = torch.empty((0,), dtype=int).to(device)
values = []

with torch.no_grad():
    for it in range(ITERS):
        print(f'iter: {it}')

        start_time()
        pos_samples = 250
        res = arm.length/float(pos_samples)
        print(res)
        quit()
        half_res = res/2.0

        # q_rand = np.random.uniform(-np.pi,np.pi,(SAMPLES,arm.n))
        # Js = arm.batch_jac(q_rand)
        # sings = np.linalg.svd(Js, compute_uv=False)
        # min_sings = np.amin(sings,axis=1)
        q_rand = ((torch.rand((SAMPLES,arm.n))-0.5)*2.0*np.pi).to(device)
        Js = arm.batch_jac(q_rand)
        sings = np.linalg.svd(Js.cpu().detach().numpy(), compute_uv=False)
        min_sings = torch.from_numpy(np.amin(sings,axis=1))

        fks = arm.batch_fk(q_rand)

        # xy_len = np.linalg.norm(fks[:,0:2,3],ord=2,axis=1)
        xy_len = torch.linalg.norm(fks[:,0:2,3],ord=2,dim=1)
        fks[:,0,3] = xy_len

        pos = fks[:,[0,2],3]
        pos += half_res

        p_index = (pos-(pos%res))/res
        p_index[:,1] += pos_samples

        # orn = fks[:,np.newaxis,0:3,2]
        # o_index = np.argmin(np.linalg.norm(verts-orn, ord=2, axis=2), axis=1)
        orn = fks[:,0:3,2].unsqueeze(1)
        o_index = torch.argmin(torch.linalg.norm(verts-orn, ord=2, dim=2), dim=1)

        print(pos[0])
        print(p_index[0])
        print((p_index[0]-torch.tensor([0,pos_samples]).to(device))*res)
        print(orn[0])
        print(o_index[0])
        print(verts[0,o_index[0]])
        quit()

        # inds = np.where(min_sings < 0.05)[0]
        # inds = np.where(min_sings < 0.01)[0]
        inds = torch.where(min_sings < 0.005)[0]

        list_inds = (p_index[inds,1]*v_shape) + (p_index[inds,0]*(pos_samples+pos_samples+1)*v_shape) + o_index[inds]
        # list_inds = list_inds.astype(int)
        list_inds = list_inds.to(int).to(device)

        # all_inds = np.union1d(all_inds,list_inds)
        all_inds = torch.cat((all_inds,list_inds)).unique()
        values.append(all_inds.shape[0])
        end_time('torch time:')

plt.plot([i for i in range(ITERS)], values)
plt.show()
