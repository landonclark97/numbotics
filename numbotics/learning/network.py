import numbotics.config as conf
from numbotics.utils import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = conf.TORCH_DEV


class FeedforwardNet(nn.Module):
    def __init__(
        self, layers, act_func=nn.LeakyReLU, opt=optim.AdamW, lr=0.01, loss=None
    ):
        super().__init__()
        self.model = nn.Sequential()
        for l in range(len(layers) - 2):
            self.model.add_module(
                "layer" + str(l), nn.Linear(layers[l], layers[l + 1], device=device)
            )
            self.model.add_module("activ" + str(l), act_func())
        self.model.add_module(
            "layer" + str(l + 1), nn.Linear(layers[-2], layers[-1], device=device)
        )
        self.opt = opt(self.model.parameters(), lr=lr)
        if loss is None:
            self.loss = nn.MSELoss(reduction="mean")
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
        w = net.model[layer * 2].weight.cpu().detach().numpy()
        return w
    except Exception as exc:
        logger.error("attempting to access non-existant neural network layer")
        raise IndexError("network does not contain {} layers".format(layer)) from exc


def get_lin_bias(net, layer):
    assert isinstance(net, FeedforwardNet)
    try:
        w = net.model[layer * 2].bias.cpu().detach().numpy()
        return w
    except Exception as exc:
        logger.error("attempting to access non-existant neural network layer")
        raise IndexError("network does not contain {} layers".format(layer)) from exc

