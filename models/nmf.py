#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple

class NNMF(nn.Module):

    def __init__(
                self,
                n_samples:int,
                n_components:int,
                n_features:int,
                init_weights:bool,
                W=None,
                H=None,
                ):
        super(NNMF, self).__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features

        if init_weights:
            assert isinstance(W, np.ndarray), '\nTo initialise components with W, NMF solution must be provided'
            assert isinstance(H, np.ndarray), 'To initialise coefficients with H, NMF solution must be provided\n'
            self.W = nn.Parameter(torch.from_numpy(W), requires_grad=True)
            self.H = nn.Parameter(torch.from_numpy(H), requires_grad=True)
        else:
            self.W = nn.Parameter(torch.randn(self.n_samples, self.n_components), requires_grad=True)
            self.H = nn.Parameter(torch.randn(self.n_features, self.n_components), requires_grad=True)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_hat = self.W.abs() @ self.H.T.abs()
        logits = x @ X_hat
        return X_hat, logits
