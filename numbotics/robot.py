import numbotics.logger as nlog
import numbotics.solver as nsol
import numbotics.spatial as spt
import numbotics.config as conf
gfx = None # imported if graphics are used - starts viz window even if imported otherwise

import numpy as np
import numpy.matlib
import scipy
import scipy.linalg

if conf.TORCH_AVAIL and conf.USE_TORCH:
    import torch


class Link():
    def __init__(self, params, dyn_params):
        assert params.shape[0] == 5
        self.link_type = params[0]
        self.a = params[1]
        self.alpha = params[2]
        self.d = params[3]
        self.theta = params[4]
        self.link_go = None

        if self.link_type == 0:
            self.base_fk = np.array([[np.cos(self.theta), -np.cos(self.alpha)*np.sin(self.theta),
                                      np.sin(self.alpha)*np.sin(self.theta), self.a*np.cos(self.theta)],
                                     [np.sin(self.theta), np.cos(self.alpha)*np.cos(self.theta),
                                      -np.sin(self.alpha)*np.cos(self.theta), self.a*np.sin(self.theta)],
                                     [0.0, np.sin(self.alpha), np.cos(self.alpha), 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])

        elif self.link_type == 1:
            self.base_fk = np.array([[1.0, np.cos(self.alpha), np.sin(self.alpha), self.a],
                                     [1.0, np.cos(self.alpha), np.sin(self.alpha), self.a],
                                     [0.0, np.sin(self.alpha), np.cos(self.alpha), self.d],
                                     [0.0, 0.0, 0.0, 1.0]])

        # elif self.link_type == 2:

        if dyn_params is not None:
            self.ixx = dyn_params[0,0]
            self.iyy = dyn_params[0,1]
            self.izz = dyn_params[0,2]
            self.ixy = dyn_params[1,0]
            self.iyz = dyn_params[1,1]
            self.ixz = dyn_params[1,2]
            self.comx = dyn_params[0,3]
            self.comy = dyn_params[0,4]
            self.comz = dyn_params[1,3]
            self.mass = dyn_params[1,4]


    @property
    def I(self):
        return np.array([[self.ixx, self.ixy, self.ixz],
                         [self.ixy, self.iyy, self.iyz],
                         [self.ixz, self.iyz, self.izz]])

    @property
    def com(self):
        return np.array([[self.comx, self.comy, self.comz]]).T


    def forward(self):

        ret_mat = np.copy(self.base_fk)
        assert len(ret_mat.shape) == 2

        if self.link_type == 0:
            ret_mat[0:3,3] = self.d

        elif self.link_type == 1:
            ret_mat[0,0] *= np.cos(self.theta)
            ret_mat[0,1] *= -np.sin(self.theta)
            ret_mat[0,2] *= np.sin(self.theta)
            ret_mat[0,3] *= np.cos(self.theta)

            ret_mat[1,0] *= np.sin(self.theta)
            ret_mat[1,1] *= np.cos(self.theta)
            ret_mat[1,2] *= -np.cos(self.theta)
            ret_mat[1,3] *= np.sin(self.theta)

        # elif self.link_type == 2:
        return ret_mat


    def forward_com(self):

        ret_mat = np.copy(self.base_fk)
        assert len(ret_mat.shape) == 2

        if self.link_type == 0:
            ret_mat[0:3,3] = self.com.squeeze(1)

        elif self.link_type == 1:
            ret_mat[0,0] *= np.cos(self.theta)
            ret_mat[0,1] *= -np.sin(self.theta)
            ret_mat[0,2] *= np.sin(self.theta)
            ret_mat[0,3] *= np.cos(self.theta)

            ret_mat[1,0] *= np.sin(self.theta)
            ret_mat[1,1] *= np.cos(self.theta)
            ret_mat[1,2] *= -np.cos(self.theta)
            ret_mat[1,3] *= np.sin(self.theta)

            ret_mat[0:3,3] = (ret_mat[0:3,0:3]@self.com).squeeze(1)

        # elif self.link_type == 2:
        return ret_mat


    def make_go(self, prev_mat):
        this_mat = prev_mat @ self.forward()
        if self.link_type == 0:
            self.link_go = gfx.pris_link(prev_mat, this_mat)
        elif self.link_type == 1:
            self.link_go = gfx.rev_link(prev_mat, this_mat)
        return this_mat
        # elif self.link_type == 2:


    def update_go(self, prev_mat):
        this_mat = prev_mat @ self.forward()
        if self.link_type == 0:
            gfx.trans_pris_link(self.link_go, prev_mat, this_mat)
        elif self.link_type == 1:
            gfx.rot_rev_link(self.link_go, prev_mat, this_mat)
        return this_mat


    @property
    def val(self):
        if self.link_type == 0:
            return self.d
        elif self.link_type == 1:
            return self.theta
        # elif self.link_type == 2:


    @val.setter
    def val(self, val):
        if self.link_type == 0:
            self.d = val
        elif self.link_type == 1:
            self.theta = val
        # elif self.link_type == 2:



class Robot():
    def __init__(self, filename=None, links=None, base=np.identity(4), params=None):

        # Either filename or links should be set, but not both
        assert (filename is not None) or (links is not None)
        assert (filename is None) or (links is None)

        # check correct formating of base matrix
        assert isinstance(base, np.ndarray)
        assert len(base.shape) == 2
        assert base.shape[0] == base.shape[1] == 4

        self.base = base

        self.links = []

        self.use_dyn = 0
        self.use_gfx = 0
        self.pos_dof = 0
        self.orn_dof = 0

        if filename:
            assert (isinstance(filename, str))
            rob_params = np.loadtxt(filename)
            self.use_dyn = int(rob_params[0,0])
            self.use_gfx = int(rob_params[0,1])
            self.n = int(rob_params[0,2])
            self.pos_dof = int(rob_params[0,3])
            self.orn_dof = int(rob_params[0,4])
            for l in range(self.n):
                self.links.append(Link(rob_params[l+1,:],
                                       rob_params[self.n+(2*l)+1:self.n+(2*l)+3,:] if self.use_dyn else None))

        if links:
            assert (isinstance(links, list))
            l_true = np.array([isinstance(l, Link) for l in links])
            assert np.all(l_true)
            self.links = links[:]
            self.n = len(links)
            if params:
                self.use_dyn = params['use_dyn']
                self.use_gfx = params['use_gfx']
                self.pos_dof = params['pos_dof']
                self.orn_dof = params['orn_dof']

        if self.use_gfx:
            global gfx # vegetables
            import numbotics.graphics
            gfx = numbotics.graphics
            gfx.inst_gfx(xlim=[-1.3,1.3],
                         ylim=[-1.3,1.3],
                         zlim=[-1.3,1.3])
            prev_mat = np.copy(self.base)
            for l in self.links:
                prev_mat = l.make_go(prev_mat)

        assert self.pos_dof in [0, 1, 2, 3]
        assert self.orn_dof in [0, 1, 3]
        assert (self.pos_dof+self.orn_dof) > 0


    def __del__(self):
        if self.use_gfx:
            # shhhhh, into the darkness you go mr graphics
            try:
                while True:
                    gfx.gfx_rate(60.0)
                    k = gfx.get_keys()
                    if 'esc' in k:
                        break
                gfx.kill_gfx()
            except:
                pass


    @property
    def q(self):
        return np.array([[q.val for q in self.links]]).T


    @q.setter
    def q(self, vals):
        assert isinstance(vals, np.ndarray)
        assert vals.shape[0] == len(self.links)
        for i, l in enumerate(self.links):
            l.val = float(vals[i])


    @property
    def length(self):
        ln = 0.0
        for l in self.links:
            ln += np.sqrt((l.a**2) + (l.d**2))
        return ln


    @property
    def q_rand(self):
        r = (np.random.rand(self.n,1)-0.5)*2.0*np.pi
        self.q = r


    @property
    def m(self):
        return self.pos_dof+self.orn_dof


    @property
    def fk(self):
        ret_mat = np.copy(self.base)
        for l in self.links:
            ret_mat = ret_mat@l.forward()
        return ret_mat


    @property
    def jac(self):
        # set J dimensions to 6, we'll adjust before returning
        J = np.zeros((6,self.n))
        fk_mats = np.empty((self.n+1,4,4))
        fk_mats[0,:,:] = np.copy(self.base)
        for i, l in enumerate(self.links):
            fk_mats[i+1,:,:] = fk_mats[i,:,:]@l.forward()
        on = fk_mats[-1,0:3,3]
        for i, l in enumerate(self.links):
            zi = fk_mats[i,0:3,2]
            oi = fk_mats[i,0:3,3]
            if l.link_type == 0:
                J[0:3,i] = zi
            elif l.link_type == 1:
                J[0:3,i] = np.cross(zi,(on-oi))
                J[3:6,i] = zi
            # elif l.link_type == 2:
        # resize Jacobian to represent workspace DoFs
        ind = np.concatenate((np.arange(self.pos_dof,dtype=int),np.arange(6-self.orn_dof,6,dtype=int)), dtype=int, casting='unsafe')
        return J[ind]


    @property
    def hess(self):
        J = self._jac()
        H = np.zeros((6,self.n,self.n))

        i = np.repeat(np.arange(self.n),self.n)
        j = np.tile(np.arange(self.n),self.n)

        H[0:3,i,j] = np.cross(J[3:6,np.minimum(i,j)].T,J[0:3,np.maximum(i,j)].T).T
        H[3:6,i,j] = np.cross(J[3:6,j].T,J[3:6,i].T).T

        return H


    @property
    def max_manifs(self):
        assert self.m in [2,3,6]
        return {2:2, 3:4, 6:16}[self.m]


    def d_sigma(self, J, H):
        assert (J.shape == (self.m,self.n)) and (H.shape == (6,self.n,self.n))
        U, D, V = np.linalg.svd(J)
        H = np.vstack((H[0:self.pos_dof,:,:],H[6-self.orn_dof:6,:,:]))
        grad = np.zeros((self.m,self.n))
        for i in range(self.m):
            for k in range(self.n):
                grad[i,k] = (U.T[i,np.newaxis,:]@H[:,:,k]@V[i,np.newaxis,:].T)[0,0]
        return grad


    def update_gfx(self):
        assert self.use_gfx
        prev_mat = np.copy(self.base)
        for l in self.links:
            prev_mat = l.update_go(prev_mat)


    def _fk(self, end_frame=None):
        assert ((end_frame is None) or (isinstance(end_frame,int)))
        if isinstance(end_frame,int):
            end_frame += 1
        ret_mat = np.copy(self.base)
        for l in self.links[:end_frame]:
            ret_mat = ret_mat@l.forward()
        return ret_mat


    def _fk_com(self, end_frame=None):
        assert ((end_frame is None) or (isinstance(end_frame,int)))
        ret_mat = np.copy(self.base)
        end_frame = self.n-1 if end_frame is None else end_frame
        for l in self.links[:end_frame]:
            ret_mat = ret_mat@l.forward()
        return ret_mat@self.links[end_frame].forward_com()


    def _jac(self, end_frame=None):
        assert ((end_frame is None) or (isinstance(end_frame,int)))
        # set J dimensions to 6, we'll adjust before returning
        J = np.zeros((6,self.n))
        fk_mats = np.empty((self.n+1,4,4))
        fk_mats[0,:,:] = np.copy(self.base)
        end_frame = self.n-1 if end_frame is None else end_frame
        for i, l in enumerate(self.links[:end_frame]):
            fk_mats[i+1,:,:] = fk_mats[i,:,:]@l.forward()
        on = fk_mats[end_frame+1,0:3,3]
        for i, l in enumerate(self.links[:end_frame+1]):
            zi = fk_mats[i,0:3,2]
            oi = fk_mats[i,0:3,3]
            if l.link_type == 0:
                J[0:3,i] = zi
            elif l.link_type == 1:
                J[0:3,i] = np.cross(zi,(on-oi))
                J[3:6,i] = zi
            # elif l.link_type == 2:
        return J


    def _jac_com(self, end_frame=None):
        assert ((end_frame is None) or (isinstance(end_frame,int)))
        J = np.zeros((6,self.n))
        fk_mats = np.empty((self.n+1,4,4))
        fk_mats[0,:,:] = np.copy(self.base)
        end_frame = self.n-1 if end_frame is None else end_frame
        for i, l in enumerate(self.links[:end_frame]):
            fk_mats[i+1,:,:] = fk_mats[i,:,:]@l.forward()
        fk_mats[end_frame+1,:,:] = fk_mats[end_frame,:,:]@self.links[end_frame].forward_com()
        on = fk_mats[end_frame+1,0:3,3]
        for i, l in enumerate(self.links[:end_frame+1]):
            zi = fk_mats[i,0:3,2]
            oi = fk_mats[i,0:3,3]
            if l.link_type == 0:
                J[0:3,i] = zi
            elif l.link_type == 1:
                J[0:3,i] = np.cross(zi,(on-oi))
                J[3:6,i] = zi
            # elif l.link_type == 2:
        return J


    def jac_dot(self, qdot):
        assert (qdot.shape == (self.n,1))
        return (self.hess@qdot).squeeze(2)


    def batch_fk(self, b_q):
        B = b_q.shape[0]

        if isinstance(b_q, np.ndarray):
            T = np.empty((B,4,4))
            ret_mat = np.empty((B,4,4))
            ret_mat[:,:,:] = self.base
            for i, l in enumerate(self.links):
                T[:,:,:] = l.base_fk
                if l.link_type == 0:
                    T[:,2,3] = b_q[:,i]

                elif l.link_type == 1:
                    T[:,0,0] *= np.cos(b_q[:,i])
                    T[:,0,1] *= -np.sin(b_q[:,i])
                    T[:,0,2] *= np.sin(b_q[:,i])
                    T[:,0,3] *= np.cos(b_q[:,i])

                    T[:,1,0] *= np.sin(b_q[:,i])
                    T[:,1,1] *= np.cos(b_q[:,i])
                    T[:,1,2] *= -np.cos(b_q[:,i])
                    T[:,1,3] *= np.sin(b_q[:,i])

                ret_mat = ret_mat@T
            return ret_mat

        else:
            assert conf.TORCH_AVAIL and conf.USE_TORCH
            assert isinstance(b_q, torch.Tensor)
            device = conf.TORCH_DEV
            b_q = b_q.to(device)
            T = torch.empty((B,4,4)).to(device)
            ret_mat = torch.empty((B,4,4)).to(device)
            ret_mat[:,:,:] = torch.tensor(self.base).to(device)
            for i, l in enumerate(self.links):
                T[:,:,:] = torch.tensor(l.base_fk).to(device)
                if l.link_type == 0:
                    T[:,2,3] = b_q[:,i]

                elif l.link_type == 1:
                    T[:,0,0] *= torch.cos(b_q[:,i])
                    T[:,0,1] *= -torch.sin(b_q[:,i])
                    T[:,0,2] *= torch.sin(b_q[:,i])
                    T[:,0,3] *= torch.cos(b_q[:,i])

                    T[:,1,0] *= torch.sin(b_q[:,i])
                    T[:,1,1] *= torch.cos(b_q[:,i])
                    T[:,1,2] *= -torch.cos(b_q[:,i])
                    T[:,1,3] *= torch.sin(b_q[:,i])

                ret_mat = ret_mat@T
            return ret_mat


    def batch_fk_err(self, x_d, b_q, mask=False):

        assert (len(b_q.shape) == 2) and (b_q.shape[1] == self.n)
        B = b_q.shape[0]
        if len(x_d.shape) == 3:
            assert x_d.shape[0] == b_q.shape[0]
            rep = False
        else:
            assert x_d.shape == (4,4)
            rep = True

        dx_mask = [True]*6
        if mask:
            dx_mask = [True if (i < self.pos_dof or i >= 6-self.orn_dof) else False for i in range(6)]

        if isinstance(b_q, np.ndarray):
            if rep:
                x_d = x_d[np.newaxis,...]
                x_d = np.repeat(x_d, B, axis=0)
            delta_x = np.zeros((B,6,1))
            x_a = self.batch_fk(b_q)
            delta_x[:,0:3,0] = x_d[:,0:3,3]-x_a[:,0:3,3]

            orn_a = x_a[:,0:3,0:3]
            orn_d = x_d[:,0:3,0:3]
            delta_x[:,3:6,0] = 0.5*(
                np.cross(orn_a[:,0:3,0],orn_d[:,0:3,0])+
                np.cross(orn_a[:,0:3,1],orn_d[:,0:3,1])+
                np.cross(orn_a[:,0:3,2],orn_d[:,0:3,2]))

        else:
            assert conf.TORCH_AVAIL and conf.USE_TORCH
            assert isinstance(b_q, torch.Tensor)
            device = conf.TORCH_DEV
            x_d = x_d.to(device)
            if rep:
                x_d = torch.unsqueeze(x_d,0)
                x_d = torch.repeat_interleave(x_d, B, dim=0)
            b_q = b_q.to(device)
            delta_x = torch.zeros((B,6,1)).to(device)
            x_a = self.batch_fk(b_q).to(device)
            delta_x[:,0:3,0] = x_d[:,0:3,3]-x_a[:,0:3,3]

            orn_a = x_a[:,0:3,0:3]
            orn_d = x_d[:,0:3,0:3]
            delta_x[:,3:6,0] = 0.5*(
                torch.cross(orn_a[:,0:3,0],orn_d[:,0:3,0])+
                torch.cross(orn_a[:,0:3,1],orn_d[:,0:3,1])+
                torch.cross(orn_a[:,0:3,2],orn_d[:,0:3,2]))

        return delta_x[:,dx_mask]


    def batch_jac(self, b_q, mask=True):
        B = b_q.shape[0]
        jac_mask = [True]*6
        if mask:
            jac_mask = [True if (i < self.pos_dof or i >= 6-self.orn_dof) else False for i in range(6)]

        if isinstance(b_q, np.ndarray):
            T = np.empty((B,4,4))
            J = np.zeros((B,6,self.n))
            ret_mat = np.empty((B,self.n+1,4,4))
            ret_mat[:,0,:,:] = self.base
            for i, l in enumerate(self.links):
                T[:,:,:] = l.base_fk
                if l.link_type == 0:
                    T[:,2,3] = b_q[:,i]

                elif l.link_type == 1:
                    T[:,0,0] *= np.cos(b_q[:,i])
                    T[:,0,1] *= -np.sin(b_q[:,i])
                    T[:,0,2] *= np.sin(b_q[:,i])
                    T[:,0,3] *= np.cos(b_q[:,i])

                    T[:,1,0] *= np.sin(b_q[:,i])
                    T[:,1,1] *= np.cos(b_q[:,i])
                    T[:,1,2] *= -np.cos(b_q[:,i])
                    T[:,1,3] *= np.sin(b_q[:,i])

                ret_mat[:,i+1,:,:] = ret_mat[:,i,:,:]@T

            on = ret_mat[:,-1,0:3,3]
            for i, l in enumerate(self.links):
                if l.link_type == 0:
                    J[:,0:3,i] = ret_mat[:,i,0:3,2]
                if l.link_type == 1:
                    J[:,0:3,i] = np.cross(ret_mat[:,i,0:3,2],(on-ret_mat[:,i,0:3,3]))
                    J[:,3:6,i] = ret_mat[:,i,0:3,2]

            return J[:,jac_mask,:]

        else:
            assert conf.TORCH_AVAIL and conf.USE_TORCH
            assert isinstance(b_q, torch.Tensor)
            device = conf.TORCH_DEV
            b_q = b_q.to(device)
            T = torch.empty((B,4,4)).to(device)
            J = torch.zeros((B,6,self.n)).to(device)
            ret_mat = torch.empty((B,self.n+1,4,4)).to(device)
            ret_mat[:,0,:,:] = torch.tensor(self.base).to(device)
            for i, l in enumerate(self.links):
                T[:,:,:] = torch.tensor(l.base_fk).to(device)
                if l.link_type == 0:
                    T[:,2,3] = b_q[:,i]

                elif l.link_type == 1:
                    T[:,0,0] *= torch.cos(b_q[:,i])
                    T[:,0,1] *= -torch.sin(b_q[:,i])
                    T[:,0,2] *= torch.sin(b_q[:,i])
                    T[:,0,3] *= torch.cos(b_q[:,i])

                    T[:,1,0] *= torch.sin(b_q[:,i])
                    T[:,1,1] *= torch.cos(b_q[:,i])
                    T[:,1,2] *= -torch.cos(b_q[:,i])
                    T[:,1,3] *= torch.sin(b_q[:,i])

                ret_mat[:,i+1,:,:] = ret_mat[:,i,:,:]@T

            on = ret_mat[:,-1,0:3,3]
            for i, l in enumerate(self.links):
                if l.link_type == 0:
                    J[:,0:3,i] = ret_mat[:,i,0:3,2]
                if l.link_type == 1:
                    J[:,0:3,i] = torch.cross(ret_mat[:,i,0:3,2],(on-ret_mat[:,i,0:3,3]))
                    J[:,3:6,i] = ret_mat[:,i,0:3,2]

            return J[:,jac_mask,:]


    def null(self, J):
        if self.n > self.m:
            return np.atleast_2d(np.linalg.svd(J)[2][self.m:self.n,:]).T
        else:
            nlog.warning('attempting to return null space of non-redundant robot')
            return np.zeros((self.n,1))


    def fk_err(self, x_d, mask=False):
        assert x_d.shape == (4,4)

        dx_mask = [True]*6
        if mask:
            dx_mask = [True if (i < self.pos_dof or i >= 6-self.orn_dof) else False for i in range(6)]

        delta_x = np.zeros((6,1))
        x_a = self.fk
        delta_x[0:3,0] = x_d[0:3,3]-x_a[0:3,3]

        orn_a = x_a[0:3,0:3]
        orn_d = x_d[0:3,0:3]
        delta_x[3:6,0] = 0.5*(
            np.cross(orn_a[0:3,0],orn_d[0:3,0])+
            np.cross(orn_a[0:3,1],orn_d[0:3,1])+
            np.cross(orn_a[0:3,2],orn_d[0:3,2]))

        return delta_x[dx_mask]


    def ik(self, x_d, ik_iters=500, dls_lambda=0.1, thresh=1e-8, rej_limit=50, disp=False):

        assert x_d.shape == (4,4)
        if disp:
            assert self.use_gfx

        self.q_rand

        success = False
        rej_cnt = 0
        dt = 1.0

        for iters in range(ik_iters):

            dx = self.fk_err(x_d, mask=True)
            J = self.jac
            dq = (J.T@np.linalg.inv((J@J.T)+(dls_lambda*np.eye(self.m))))@dx

            self.q += (dq*dt)

            dx_new = self.fk_err(x_d, mask=True)

            new_err = np.linalg.norm(dx_new)
            err = np.linalg.norm(dx)

            if new_err < err:
                dls_lambda /= 2.0
                rej_cnt = 0
            else:
                #self.q = q_orig
                dls_lambda *= 2.0
                dls_lambda = min(1.0,dls_lambda)
                rej_cnt += 1
                if rej_cnt >= rej_limit:
                    nlog.warning('too many consecutive rejected IK steps - hit limit: ' + str(rej_limit))
                    break

            if new_err < thresh:
                success = True
                break

            if disp:
                self.update_gfx()
                gfx.gfx_rate(1.0/dt)

        return success


    def smm(self, x_d, smm_iters=1000, step=0.05, samples=128, ik_thresh=1e-8, sing_thresh=5e-3, torus=True, squeeze=False, disp=False):

        assert x_d.shape == (4,4)
        if disp:
            assert self.use_gfx

        if np.linalg.norm(self.fk_err(x_d)) > ik_thresh:
            s_ik = self.ik(x_d)
            if not s_ik:
                return 0, None

        J = self.jac
        if np.amin(np.linalg.svd(J,compute_uv=False)) <= sing_thresh:
            ret = np.zeros((samples,self.n,1), dtype=np.complex128)
            if squeeze:
                ret = np.squeeze(ret,2)
            return 2, ret

        tr = lambda x: np.cos(x) + 1j*np.sin(x)
        tr_inv = lambda x: np.arctan2(np.imag(x),np.real(x))

        success = 0

        orig_nl_vec = self.null(J)
        prev_nl_vec = np.copy(orig_nl_vec)

        orient = np.zeros((self.n,1))
        for i in range(self.n):
            orient[i,0] = ((-1.0)**(i+1))*np.linalg.det(np.delete(J,i,axis=1))

        if (orig_nl_vec.T@orient)[0,0] < 0:
            orig_nl_vec = -orig_nl_vec
            prev_nl_vec = -prev_nl_vec

        start_q = np.copy(self.q)

        smm = np.empty((1,self.n,1))
        smm[0,:,:] = np.copy(start_q)

        for iters in range(smm_iters):

            J = self.jac

            sig = np.min(np.linalg.svd(J)[1])
            if sig <= sing_thresh:
                ret = np.zeros((samples,self.n,1), dtype=np.complex128)
                if squeeze:
                    ret = np.squeeze(ret,2)
                return 2, ret

            nl_vec = self.null(J)
            if (nl_vec.T @ prev_nl_vec)[0,0] < 0:
                nl_vec = -nl_vec
            prev_nl_vec = np.copy(nl_vec)

            x_err = self.fk_err(x_d, mask=True)

            dq = nl_vec + np.linalg.pinv(J)@x_err
            dq = (dq/np.linalg.norm(dq))*step

            self.q += dq
            smm = np.append(smm,[self.q],axis=0)

            err = tr_inv(tr(start_q)/tr(self.q))
            if np.linalg.norm(err) < (9.0/5.0)*step and iters > 4:
                success = 1
                break

            if disp:
                self.update_gfx()
                gfx.gfx_rate(1.0/step)

        smm_int = np.empty((samples,self.n,1))
        for l in range(self.n):
            smm_int[:,l,0] = np.interp(np.linspace(0,smm.shape[0]-1,samples,endpoint=True),
                                       np.linspace(0,smm.shape[0]-1,smm.shape[0],endpoint=True),
                                       smm[:,l,0])
        smm_int = tr_inv(tr(smm_int))
        if squeeze:
            smm_int = np.squeeze(smm_int,2)
        if not torus:
            smm_int = tr_inv(smm_int)

        return success, smm_int


    def all_smms(self, x_d, samples=128, step=0.05, sing_thresh=5e-3, smm_iters=1000, without_new=25, max_fails=5, diff_thresh=0.15, plot=False):

        assert x_d.shape == (4,4)
        if plot:
            assert conf.MATPLOT_AVAIL

        tr = lambda t: np.cos(t) + 1j*np.sin(t)
        tr_inv = lambda t: np.arctan2(np.imag(t),np.real(t))

        max_manifs = self.max_manifs
        current_without = 0
        fails = 0
        success = 0

        manifold_data = np.empty((0,self.n,samples),dtype=np.complex128)

        while current_without < without_new:

            self.q_rand
            s = self.ik(x_d)

            if s:
                fails = 0
                s_smm, angles = self.smm(x_d, samples=samples, smm_iters=smm_iters, step=step, sing_thresh=sing_thresh, squeeze=True, disp=False)

                if s_smm == 1:
                    ang = tr(angles)
                    add_manifold = True
                    for mf in manifold_data:
                        if np.min(np.linalg.norm(tr_inv(mf.T/ang[np.random.randint(0,samples),:]),axis=1)) < diff_thresh:
                            add_manifold = False
                            current_without += 1
                            break

                    if add_manifold:
                        current_without = 0
                        manifold_data = np.concatenate((manifold_data, np.swapaxes(ang,0,1)[np.newaxis,...]), axis=0)

                elif s_smm == 2:
                    manifold_data = np.zeros((max_manifs+1,self.n,samples,0),dtype=np.complex128)
                    success = 2

                else:
                    current_without += 1

                if manifold_data.shape[0] > max_manifs:
                    break

            else:
                fails += 1
                if fails > max_fails:
                    break

        if (manifold_data.shape[0] != 0) and manifold_data.shape[0] <= max_manifs:
            success = 1
            if plot:
                import matplotlib.pyplot as plt
                plots = ((self.n-2)//2)+1
                fig = plt.figure()
                axs = [fig.add_subplot(plots,1,i+1,projection='3d') for i in range(plots)]
                for i in range(plots):
                    x_ind = min((i*2),self.n-1)
                    y_ind = min((i*2)+1,self.n-1)
                    z_ind = min((i*2)+2,self.n-1)
                    for mf in manifold_data:
                        axs[i].scatter(tr_inv(mf[x_ind,:]),tr_inv(mf[y_ind,:]),tr_inv(mf[z_ind,:]))
                        axs[i].set_xlim(-np.pi,np.pi)
                        axs[i].set_ylim(-np.pi,np.pi)
                        axs[i].set_zlim(-np.pi,np.pi)
                plt.show()

        return success, manifold_data


    def sim(self, T, dynamics='nonlinear', sol=nsol.ARK23(), u_func=None, disp=False):

        if disp:
            assert self.use_gfx
        assert dynamics in ['nonlinear', 'linearized']

        if dynamics == 'linearized':
            state_mat = np.zeros((self.n*2,self.n*2))
            state_mat[0:self.n,self.n:self.n*2] = np.identity(self.n)
            state_mat[self.n:self.n*2,0:self.n] = -1.0*np.identity(self.n)
            state_mat[self.n:self.n*2,self.n:self.n*2] = -2.0*np.identity(self.n)

            if u_func is None:
                f = lambda t, x: state_mat@x
            else:
                f = lambda t, x: state_mat@x + u_func(x,t)

        else:
            if u_func is None:
                def f(t,x):
                    pass
            else:
                def f(t,x):
                    pass

        t = 0.0
        dt = 1.0/60.0

        q = np.zeros((self.n*2,1))
        q[0:self.n] = self.q

        while t < T:

            dt_a, q = sol.step(f,t,q,t+dt)
            self.q = q[0:self.n]

            t += dt_a

            if disp:
                assert self.use_gfx
                self.update_gfx()
                gfx.gfx_rate(1.0/dt_a)


    def sim_zoh(self, T, sol=nsol.ARK23(), u_func=lambda x,t: np.zeros((x.shape[0],1)), control_freq=100.0, disp=False):
        if disp:
            assert self.use_gfx
        assert (control_freq > 0.0) and (u_func is not None)

        state_mat = np.zeros((self.n*2,self.n*2))
        state_mat[0:self.n,self.n:self.n*2] = np.identity(self.n)
        state_mat[self.n:self.n*2,0:self.n] = -1.0*np.identity(self.n)
        state_mat[self.n:self.n*2,self.n:self.n*2] = -2.0*np.identity(self.n)

        t = 0.0
        dt = 1.0/60.0

        q = np.zeros((self.n*2,1))
        q[0:self.n] = self.q
        qdot = np.zeros((self.n*2,1))
        u = np.zeros((self.n*2,1))

        control_x = np.copy(q)
        control_t = np.copy(t)

        def f(t, x):
            nonlocal control_x
            nonlocal control_t
            if (t - control_t) >= 1.0/control_freq:
                control_t = np.copy(t)
                control_x = np.copy(x)
            return state_mat@x + u_func(control_x, control_t)

        while t < T:

            dt_a, q = sol.step(f,t,q,t+dt)
            self.q = q[0:self.n]

            t += dt_a

            if disp:
                assert self.use_gfx
                self.update_gfx()
                gfx.gfx_rate(1.0/dt_a)






if __name__ == '__main__':

    # s_ik = r.ik(x_d, disp=True)

    r = Robot(filename='test.rob')

    def u(x,t):
        control = np.zeros((x.shape[0],1))
        control[x.shape[0]:2*x.shape[0]] = -x[x.shape[0]:2*x.shape[0]]
        return control

    r.q = np.array([[np.pi/2.0,-np.pi/2.0,np.pi,np.pi,-np.pi,np.pi,-np.pi]]).T
    r.sim_zoh(np.inf,sol=nsol.ARK23(auto_inc=True),u_func=u,control_freq=10.0,disp=True)




    x_d = np.array([[0.0403,0.0,0.3433,np.pi,np.pi/2.0,0.0]]).T
    s, smm = r.smm(x_d, squeeze=True, disp=False)


    del r
