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

class NMFTrainer(object):

    def __init__(
                self,
                nmf:Any,
                optim:Any,
                X:torch.Tensor,
                lr:float,
                temperature:torch.Tensor,
                max_epochs:int,
                task:str,
                device:torch.device,
                verbose:bool=True,
                ):

    self.nmf = nmf
    self.optim = optim
    self.X = X
    self.lr = lr
    self.temperature = temperature
    self.max_epochs = max_epochs
    self.task = task
    self.device = device
    self.verbose = verbose

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

    def fit(
            self,
            train_batches:Iterator[torch.Tensor],
            val_batches:Iterator[torch.Tensor],
            ):
        self.nmf.train()
        rerrors = []
        centropies = []
        losses = []
        accuracies = []
        for epoch in range(epochs):
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

    return train_losses, train_accs, val_losses, val_accs
