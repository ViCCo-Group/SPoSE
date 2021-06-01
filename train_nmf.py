#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import random
import os
import torch
import utils

import copy
import numpy as np

from collections import defaultdict
from model.nmf import NNMF
from nmf_optimization import NMFTrainer
from typing import List, Tuple

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--optimizer', type=str,
        choices=['SGD', 'RMSprop', 'Adam', 'AdamW'])
    aa('--learning_rate', type=float, default=0.001,
        help='step size multiplied by batch-averaged gradients during each iteration')
    aa('--batch_size', metavar='B', type=int, default=10)
    aa('--max_epochs', metavar='T', type=int, default=100,
        help='maximum number of epochs to optimize VSPoSE for')
    aa('--num_samples', metavar='N', type=int,
        help='number of samples to be drawn')
    aa('--criteria', type=str, nargs='+',
        help='list of different convergence criteria to be tested')
    aa('--window_size', type=int, default=50,
        help='window size to be used for checking convergence criterion with linear regression')
    aa('--steps', type=int,
        help='perform validation and save model parameters every <steps> epochs')
    aa('--results_dir', type=str,
        help='path/to/results')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def run(
        optimizer:str,
        lr:float,
        batch_size:int,
        max_epochs:int,
        num_samples:int,
        criteria:List[str],
        window_size:int,
        steps:int,
        device:torch.device,
        results_dir:str,
) -> None:
    #load triplets into memory
    train_triplets, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir)
    n_items = utils.get_nitems(train_triplets)
    #load train and test mini-batches
    train_batches, val_batches = utils.load_batches(
                                                  train_triplets=train_triplets,
                                                  test_triplets=test_triplets,
                                                  n_items=n_items,
                                                  batch_size=batch_size,
                                                  sampling_method='normal',
                                                  rnd_seed=rnd_seed,
                                                  )
    results = defaultdict(dict)
    models = copy_model(model, criteria)

    nmf_cd = NMF(n_components=n_components, init='nndsvd', max_iter=5000, random_state=rnd_seed)
    W_nmf_cd = nmf_cd.fit_transform(X)
    H_nmf_cd = nmf_cd.components_
    nmf_gd = NNMF(n_samples=X.shape[0], n_components=n_components, n_features=X.shape[1], init_weights=True, W=W_nmf_cd, H=H_nmf_cd)
    optim = Adam(nmf_gd.parameters(), lr=lr)
    trainer = NMFTrainer(
                         nmf=nmf_gd,
                         optim=optim,
                         X=X,
                         lr=lr,
                         temperature=temperature,
                         epochs=epochs,
                         task=task,
                         device=device,
                         verbose=verbose,
                         )
        model = models[k]
        trainer = Trainer(
                         lr=lr,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         steps=steps,
                         criterion=criterion,
                         optimizer=optimizer,
                         device=device,
                         results_dir=results_dir,
                         window_size=window_size,
        )
        steps, train_losses, val_losses = trainer.fit(model=model, train_batches=train_batches, val_batches=val_batches, verbose=True)
        print(f'Finished optimization for {criterion} criterion after {steps} steps\n')

        results[criterion]['stopping'] = steps
        results[criterion]['train_losses'] = train_losses
        results[criterion]['val_losses'] = val_losses

    _save_results(trainer, results)

def _save_results(trainer:object, results:dict) -> None:
    with open(os.path.join(trainer.PATH, 'results.txt'), 'wb') as f:
        f.write(pickle.dumps(results))

if __name__ == '__main__':
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    device = torch.device(args.device)
    run(
        optimizer=args.optimizer,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        criteria=args.criteria,
        window_size=args.window_size,
        steps=args.steps,
        device=device,
        results_dir=args.resul
