#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['NeuralNMF', 'BatchNMF']

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple

class NeuralNMF(nn.Module):

    def __init__(
                self,
                n_samples:int,
                n_components:int,
                n_features:int,
                init_weights:bool,
                W=None,
                H=None,
    ):
        super(NeuralNMF, self).__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features
        self.W = nn.Linear(self.n_samples, self.n_components, bias=False)

        if init_weights:
            assert isinstance(W, np.ndarray), '\nTo initialise components with W, NMF solution must be provided'
            assert isinstance(H, np.ndarray), 'To initialise coefficients with H, NMF solution must be provided\n'
            self.W.weight.data = torch.from_numpy(W.T).type(self.W.weight.data.dtype)
            self.H = nn.Parameter(torch.from_numpy(H.T).type(self.W.weight.data.dtype), requires_grad=True)
        else:
            self.H = nn.Parameter(torch.randn(self.n_features, self.n_components), requires_grad=True)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.W.weight.data.abs_()
        logits = self.W(x)
        return logits

class BatchNMF(nn.Module):

    def __init__(
                self,
                n_samples:int,
                n_components:int,
                n_features:int,
                init_weights:bool,
                W=None,
                H=None,
                ):
        super(BatchNMF, self).__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features

        if init_weights:
            assert isinstance(W, np.ndarray), '\nTo initialise components with W, NMF solution must be provided'
            assert isinstance(H, np.ndarray), 'To initialise coefficients with H, NMF solution must be provided\n'
            self.W = nn.Parameter(torch.from_numpy(W).type(torch.float), requires_grad=True)
            self.H = nn.Parameter(torch.from_numpy(H.T).type(torch.float), requires_grad=True)
        else:
            self.W = nn.Parameter(torch.randn(self.n_samples, self.n_components), requires_grad=True)
            self.H = nn.Parameter(torch.randn(self.n_features, self.n_components), requires_grad=True)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.W.abs()
        return logits
