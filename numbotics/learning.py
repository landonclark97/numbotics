import numbotics.config as conf
import numbotics.logger as nlog

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = conf.TORCH_DEV

class FeedforwardNet(nn.Module):
    def __init__(self, layers, act_func=nn.LeakyReLU, opt=optim.AdamW, lr=0.01, loss=None):
        super().__init__()
        self.model = nn.Sequential()
        for l in range(len(layers)-2):
            self.model.add_module('layer'+str(l), nn.Linear(layers[l],layers[l+1], device=device))
            self.model.add_module('activ'+str(l), act_func())
        self.model.add_module('layer'+str(l+1), nn.Linear(layers[-2],layers[-1], device=device))
        self.opt = opt(self.model.parameters(), lr=lr)
        if loss is None:
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = loss

    def forward(self, x):
        return self.model(x.to(device))

    def train(self, x, y):
        self.opt.zero_grad()
        loss = self.loss(y.to(device), self.model(x.to(device)).to(device))
        loss.backward()
        self.opt.step()
        return loss.item()

    def save(self, path, del_opt=False):
        if del_opt:
            del self.opt
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path, map_location=device, weights_only=False)


def get_lin_weight(net, layer):
    assert isinstance(net, FeedforwardNet)
    try:
        w = net.model[layer*2].weight.cpu().detach().numpy()
        return w
    except Exception as exc:
        nlog.error('attempting to access non-existant neural network layer')
        raise IndexError('network does not contain {} layers'.format(layer)) from exc


def get_lin_bias(net, layer):
    assert isinstance(net, FeedforwardNet)
    try:
        w = net.model[layer*2].bias.cpu().detach().numpy()
        return w
    except Exception as exc:
        nlog.error('attempting to access non-existant neural network layer')
        raise IndexError('network does not contain {} layers'.format(layer)) from exc



'''
import torch

E = 2000

D = 100000
X = torch.empty((D,r.n))
Y = torch.empty((D,1))
for i in range(D):
    r.q_rand
    X[i,:] = torch.tensor(r.q.T)
    Y[i,0] = torch.tensor(np.linalg.cond(r.jac))/100.0
    print(f'\riteration: {i}, cond: {Y[i,0]}', end='')
print()

model = lrn.SingularNet(r.n)
for e in range(E):
    l = model.update(X,Y)
    print(f'\repoch: {e}, loss: {l}', end='')
print()

# r.q_rand
r.q = np.zeros((7,1))
print(model(r.q.T))

for i in range(1000):
    J = r.jac
    print(np.linalg.cond(J))
    v = (np.identity(r.n) - (np.linalg.pinv(J)@J))@model(r.q.T)
    v = v/np.linalg.norm(v)
    r.q -= 0.1*v
    print(np.linalg.cond(r.jac))
    r.update_gfx()
    gfx.gfx_rate(60.0)

'''
