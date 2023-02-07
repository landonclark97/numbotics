import numbotics.config as conf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = conf.TORCH_DEV


class SingularNet(nn.Module):
    def __init__(self, n):
        super(SingularNet, self).__init__()
        self.l1 = nn.Linear(n,60).to(device)
        self.l2 = nn.Linear(60,100).to(device)
        self.l3 = nn.Linear(100,100).to(device)
        self.l4 = nn.Linear(100,1).to(device)
        self.opt = optim.Adam(self.parameters(),lr=0.005)

    def __call__(self, x):
        self.opt.zero_grad()
        x_ret = torch.tensor(x,dtype=torch.float32).requires_grad_(True)
        self.forward(x_ret.to(device)).backward()
        grad = x_ret.grad
        self.opt.zero_grad()
        return grad.detach().numpy().T

    def parameters(self):
        return [self.l1.weight, self.l1.bias,
                self.l2.weight, self.l2.bias,
                self.l3.weight, self.l3.bias,
                self.l4.weight, self.l4.bias]

    def forward(self, x):
        out = x.to(device)
        out = F.leaky_relu(self.l1(out))
        out = F.leaky_relu(self.l2(out))
        out = F.leaky_relu(self.l3(out))
        return self.l4(out)

    def update(self, X, Y):
        Yhat = self.forward(X)
        loss = torch.mean(torch.mean(torch.pow(Yhat-Y.to(device),2),dim=1),dim=0)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()





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
