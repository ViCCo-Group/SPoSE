#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'SPoSE',
            'l1_regularization',
            ]

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

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
