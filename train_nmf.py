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
from model.nmf import NeuralNMF, BatchNMF
from nmf_optimization import NMFTrainer
from typing import List, Tuple

os.environ['PYTHONIOENCODING']='UTF-8'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['OMP_NUM_THREADS']='1' #number of cores used per Python process (set to 2 if HT is enabled, else keep 1)

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets')
    aa('--in_path', type=str,
        help='path to models (this is equal to the path where model weights were stored at the end of training)')
    aa('--out_path', type=str,
        help='path where to store final nmf components matrix')
    aa('--learning_rate', type=float, default=0.001,
        help='learning rate to be used in optimizer')
    aa('--optimizer', type=str,
        choices=['SGD', 'RMSprop', 'Adam', 'AdamW'])
    aa('--criterion', type=str,
        choices=['eb', 'val'],
        help='criterion for early stopping')
    aa('--n_components', type=int, nargs='+',
        help='list of component values to run grid search over (note that number of component values determines the number of initialized Python processes)')
    aa('--out_format', type=str,
        choices=['mat', 'txt', 'npy'],
        help='format in which to store nmf weights matrix to disk')
    aa('--batch_size', metavar='B', type=int, default=100,
        choices=[16, 25, 32, 50, 64, 100, 128, 150, 200, 256],
        help='number of triplets in each mini-batch')
    aa('--epochs', metavar='T', type=int, default=500,
        help='maximum number of epochs to optimize NMF for')
    aa('--window_size', type=int, default=None,
        help='window size to be used for checking convergence criterion with linear regression (iff criterion is validation)')
    aa('--device', type=str, default='cpu',
        choices=['cpu'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args


def run(
        process_id:int,
        task:str,
        triplets_dir:str,
        in_path:str,
        out_path:str,
        lr:float,
        optimizer:str,
        criterion:str,
        n_components:List[int],
        out_format:str,
        epochs:int,
        device:torch.device,
        init_weights:bool=False,
        window_size:int=None,
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

    p = n_components[process_id]

    if init_weights:
        nmf_cd = NMF(n_components=p, init='nndsvd', max_iter=5000, random_state=rnd_seed)
        W_nmf_cd = nmf_cd.fit_transform(X)
        H_nmf_cd = nmf_cd.components_
    else:
        W_nmf_cd = None
        H_nmf_cd = None

    if criterion == 'eb':
        nmf_gd = NeuralNMF(n_samples=X.shape[0], n_components=n_components, n_features=X.shape[1], init_weights=init_weights, W=W_nmf_cd, H=H_nmf_cd)
    else:
        nmf_gd = BatchNMF(n_samples=X.shape[0], n_components=n_components, n_features=X.shape[1], init_weights=init_weights, W=W_nmf_cd, H=H_nmf_cd)

    

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
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    device = torch.device(args.device)

    n_subprocs = len(args.n_components)
    if n_subprocs > os.cpu_count()-1:
        raise Exception('Number of initialized processes exceeds the number of available CPU cores.')
    print(f'\nUsing {n_subprocs} CPU cores for parallel training\n')
    device = torch.device(args.device)

     torch.multiprocessing.spawn(
        run,
        args=(
        args.task,
        args.triplets_dir,
        args.in_path,
        args.out_path,
        args.learning_rate,
        args.optimizer,
        args.criterion,
        args.n_components,
        args.out_format,
        args.batch_size,
        args.epochs,
        args.window_size,
        device
        ),
        nprocs=n_subprocs,
        join=True)
