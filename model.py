#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'SPoSE',
            'VSPoSE',
            'l1_regularization',
            ]

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.laplace import Laplace
from typing import Tuple

class VSPoSE(nn.Module):

    def __init__(
                self,
                in_size:int,
                out_size:int,
                eps:float=1e-7,
                ):
        super(VSPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.eps = eps
        #NOTE: in the end we want to extract the model parameters W_mu and W_b to sample objects offline
        self.encoder_mu = nn.Linear(self.in_size, self.out_size, bias=True)
        self.encoder_b = nn.Linear(self.in_size, self.out_size, bias=True)
        self.decoder = nn.Linear(self.out_size, self.out_size, bias=True)

    def reparameterize(self, mu:torch.Tensor, b:torch.Tensor, device:torch.device) -> torch.Tensor:
        laplace = Laplace(loc=torch.zeros(b.size()), scale=torch.ones(b.size()))
        U = laplace.sample().to(device) #draw random sample from a standard Laplace distribution with mu = 0 and lam = 1
        z = U.mul(b).add(mu) #perform reparameterization trick
        z = z.abs() #apply absolute value to impose non-negativity constraint on z
        return z

    def forward(self, x:torch.Tensor, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.encoder_mu(x) #mu = mode = loc param
        b = self.encoder_b(x) #b = scale param
        b = b.clamp(min=self.eps) #b is constrained to be in the positive real number space R+ (b > 0)
        z = self.reparameterize(mu, b, device)
        logits = self.decoder(z)
        l = (b.log()*-1.0).exp() #this is equivalent to but numerically more stable than b.pow(-1)
        return logits, z, mu, l

class SPoSE(nn.Module):

    def __init__(
                self,
                in_size:int,
                out_size:int,
                init_weights:bool=True,
                ):
        super(SPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def _initialize_weights(self) -> None:
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

def l1_regularization(model) -> torch.Tensor:
    l1_reg = torch.tensor(0., requires_grad=True)
    for n, p in model.named_parameters():
        if re.search(r'weight', n):
            l1_reg = l1_reg + torch.norm(p, 1)
    return l1_reg
