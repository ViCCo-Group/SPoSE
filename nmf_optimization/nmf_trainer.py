#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['NMFTrainer']

import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Iterator, Tuple, Any
from scipy.stats import linregress

class NMFTrainer(object):

    def __init__(
                self,
                nmf:Any,
                optim:str,
                X:torch.Tensor,
                lr:float,
                temperature:torch.Tensor,
                epochs:int,
                batch_size:int,
                task:str,
                criterion:str,
                device:torch.device,
                window_size=None,
                verbose:bool=True,
                ):

    self.nmf = nmf
    self.optim = optim
    self.X = X
    self.lr = lr
    self.temperature = temperature
    self.epochs = epochs
    self.batch_size = batch_size
    self.task = task
    self.criterion = criterion
    self.device = device
    self.verbose = verbose

    if self.criterion == 'validation':
        assert isinstance(window_size, int), '\nWindow size parameter is required to examine convergence criterion\n'
        self.window_size = window_size

    def get_optim(self):
        if self.optim == 'SGD':
            optim = SGD(self.nmf.parameters(), lr=self.lr)
        elif self.optim == 'RMSprop':
            optim = RMSprop(self.nmf.parameters(), lr=self.lr, alpha=0.99, eps=1e-08)
        elif self.optim == 'Adam':
            optim = Adam(self.nmf.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            optim = AdamW(self.nmf.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        return optim

    def squared_norm(self, X_hat) -> torch.Tensor:
        return torch.norm(self.X - X_hat, p='fro').pow(2)

    def validation(self, val_batches:Iterator[torch.Tensor]) -> Tuple[float, float]:
        self.nmf.eval()
        with torch.no_grad():
            batch_losses_val = torch.zeros(len(val_batches))
            batch_accs_val = torch.zeros(len(val_batches))
            for i, batch in enumerate(val_batches):
                batch = batch.to(self.device)
                _, logits = self.nmf(batch)
                anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)
                val_centropy = utils.trinomial_loss(anchor, positive, negative, self.task, self.temperature)
                val_acc = utils.choice_accuracy(anchor, positive, negative, self.task)
                batch_losses_val += val_loss.item()
                batch_accs_val += val_acc
        avg_val_loss = batch_losses_val.mean().item()
        avg_val_acc = batch_accs_val.mean().item()
        return avg_val_loss, avg_val_acc

    def register_hooks(neuralnmf):
        """register a backward hook to store per-sample gradients"""
        for m in neuralnmf.modules():
            m.register_forward_hook(collect_acts)
            m.register_backward_hook(collect_grads)
        return neuralnmf

    def collect_acts(layer, input, output):
        """store per-sample gradients (weights are shared between encoders)"""
        setattr(layer, 'inputs', input[0].detach())

    def collect_grads(layer, input, output):
        """store per-sample gradients (weights are shared between encoders)"""
        if not hasattr(layer, 'gradients'):
            setattr(layer, 'gradients', [])
        layer.gradients.append(output[0].detach())

    def clear_backprops(neuralnmf:nn.Module) -> None:
        """remove gradient information in every layer"""
        for m in neuralnmf.modules():
            if hasattr(m, 'gradients'):
                del m.gradients

    def compute_sample_grads(neuralnmf) -> None:
        for n, m in neuralnmf.named_modules():
            if n == 'W':
                A = m.inputs
                M = A.shape[0]
                B = m.gradients[0] * M
                setattr(m.weight, 'sample_gradients', torch.einsum('ni,nj->nij', B, A))

    def estimate_grad_var(sample_gradients, average_gradients) -> torch.Tensor:
        return (sample_gradients - average_gradients[None, ...]).pow(2).sum(dim=0)/(sample_gradients.shape[0]-1)

    def eb_criterion(avg_grad:torch.Tensor, var_estimator:torch.Tensor) -> float:
        D = avg_grad.shape[0] * avg_grad.shape[1]
        return 1 - (self.batch_size/D)*((avg_grad.pow(2)/var_estimator).sum())

    def get_means_and_vars(neuralnmf) -> Tuple[torch.Tensor, torch.Tensor]:
        avg_grad = neuralnmf.W.weight.grad
        var_estimator = estimate_grad_var(neuralnmf.W.weight.sample_gradients, neuralnmf.W.weight.grad)
        return avg_grad, var_estimator

    def fit(
            self,
            train_batches:Iterator[torch.Tensor],
            val_batches:Iterator[torch.Tensor],
            ):
        if self.criterion == 'eb':
            self.nmf = self.register_hooks(self.nmf)
            stop_training = False
        self.nmf.train()
        optim = self.get_optim()
        rerrors = []
        centropies = []
        losses = []
        accuracies = []
        for epoch in range(self.epochs):
            batch_rerrors = torch.zeros(len(train_batches))
            batch_centropies = torch.zeros(len(train_batches))
            batch_losses = torch.zeros(len(train_batches))
            batch_accs = torch.zeros(len(train_batches))
            for i, batch in enumerate(train_batches):
                self.optim.zero_grad()
                batch = batch.to(self.device)
                X_hat, logits = self.nmf(batch)
                anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, embed_dim)), dim=1)
                c_entropy = utils.trinomial_loss(anchor, positive, negative, self.task, self.temperature)
                r_error = self.squared_norm(X_hat)
                loss = alpha * r_error + c_entropy
                loss.backward()
                self.optim.step()

                if self.criterion == 'eb':
                    self.compute_sample_grads(self.nmf)
                    avg_grad, var_estimator = self.get_means_and_vars(self.nmf)
                    eb_criterion = self.eb_criterion_(avg_grad, var_estimator)
                    self.clear_backprops(self.nmf)
                    if eb_criterion > 0:
                        stop_training = True

                batch_rerrors[i] += r_error.item()
                batch_centropies[i] += c_entropy.item()
                batch_losses[i] += loss.item()
                batch_accs[i] += utils.choice_accuracy(anchor, positive, negative, self.task)
                iter += 1

            avg_rerror = batch_rerrors.mean().item()
            avg_centropy = batch_centropies.mean().item()
            avg_loss = batch_losses.mean().item()
            avg_acc = batch_accs.mean().item()

            rerrors.append(avg_rerror)
            centropies.append(avg_centropy)
            losses.append(avg_loss)
            accuracies.append(avg_acc)

            avg_val_loss, avg_val_acc = self.validation(val_batches)
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)

            if self.verbose:
                print("\n==============================================================================================================")
                print(f'====== Epoch: {epoch+1}, Train acc: {avg_acc:.3f}, Val acc: {avg_val_acc:3.f}, Cross-entropy loss: {avg_centropy:.3f}, /
                Reconstruction error: {avg_rerror:.3f}  ======')
                print("==============================================================================================================\n")

            if criterion == 'eb':
                if stop_training:
                    break
            else:
                if (epoch + 1) > self.window_size:
                    #check termination condition (we want to train until convergence)
                    lmres = linregress(range(self.window_size), val_losses[(epoch + 1 - self.window_size):(epoch + 2)])
                    if (lmres.slope > 0) or (lmres.pvalue > .1):
                        break

    return train_losses, train_accs, val_losses, val_accs
