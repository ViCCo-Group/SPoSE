#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import random
import os
import torch
import utils
import json
import scipy

import copy
import numpy as np

from collections import defaultdict
from sklearn.decomposition import NMF
from models.nmf import *
from nmf_optimization.nmf_trainer import NMFTrainer
from typing import List, Tuple

os.environ['PYTHONIOENCODING']='UTF-8'
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
        choices=['eb', 'train'],
        help='criterion for early stopping')
    aa('--alpha', type=float,
        help='scaling factor for reconstruction error')
    aa('--n_components', type=int, nargs='+',
        help='list of component values to run grid search over (note that number of component values determines the number of initialized Python processes)')
    aa('--init_weights', action='store_true',
        help='whether to initialise weights of gradient-based NMF with weights of standard NMF solution')
    aa('--out_format', type=str,
        choices=['mat', 'txt', 'npy'],
        help='format in which to store nmf weights matrix to disk')
    aa('--batch_size', metavar='B', type=int, default=100,
        choices=[16, 25, 32, 50, 64, 100, 128, 150, 200, 256],
        help='number of triplets in each mini-batch')
    aa('--epochs', metavar='T', type=int, default=500,
        help='maximum number of epochs to optimize NMF for')
    aa('--verbose', action='store_true')
    aa('--window_size', type=int, default=None,
        help='window size to be used for checking convergence criterion with linear regression (iff criterion is validation)')
    aa('--device', type=str, default='cpu',
        choices=['cpu'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def get_weights(PATH:str) -> List[np.ndarray]:
    weights = []
    for root, _, files in os.walk(PATH):
        for file in files:
            if file == 'weights_sorted.npy':
                with open(os.path.join(root, file), 'rb') as f:
                    W = utils.remove_zeros(np.load(f).T).T
                weights.append(W)
    return weights

def run(
        process_id:int,
        task:str,
        triplets_dir:str,
        in_path:str,
        out_path:str,
        lr:float,
        alpha:float,
        optimizer:str,
        criterion:str,
        n_components:List[int],
        out_format:str,
        batch_size:int,
        epochs:int,
        rnd_seed:int,
        device:torch.device,
        init_weights:bool=False,
        window_size:int=None,
        verbose:bool=True,
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
    weights = get_weights(in_path)
    X = np.hstack(weights)
    p = n_components[process_id]

    if init_weights:
        nmf_cd = NMF(n_components=p, init=None, max_iter=5000, random_state=rnd_seed)
        W_nmf_cd = nmf_cd.fit_transform(X)
        H_nmf_cd = nmf_cd.components_
    else:
        W_nmf_cd = None
        H_nmf_cd = None

    if criterion == 'eb':
        nmf_gd = NeuralNMF(n_samples=X.shape[0], n_components=p, n_features=X.shape[1], init_weights=init_weights, W=W_nmf_cd, H=H_nmf_cd)
    else:
        #nmf_gd = BatchNMF(n_samples=X.shape[0], n_components=p, n_features=X.shape[1], init_weights=init_weights, W=W_nmf_cd, H=H_nmf_cd)
        nmf_gd = FrozenNMF(n_samples=X.shape[0], n_components=p, n_features=X.shape[1], W=W_nmf_cd, H=H_nmf_cd)

    X = torch.from_numpy(X)
    nmf_gd.to(device)
    nmf_trainer = NMFTrainer(
                             nmf=nmf_gd,
                             optim=optimizer,
                             X=X,
                             lr=lr,
                             alpha=alpha,
                             temperature=torch.tensor(1.),
                             epochs=epochs,
                             batch_size=batch_size,
                             task=task,
                             criterion=criterion,
                             device=device,
                             window_size=window_size,
                             verbose=verbose,
                             )
    train_losses, train_accs, val_losses, val_accs = nmf_trainer.fit(process_id, train_batches, val_batches)

    results = {}
    results['val_acc'] = val_accs[-1]
    results['val_loss'] = val_losses[-1]
    results['train_acc'] = train_accs[-1]
    results['train_losses'] = train_losses[-1]

    dir_list = triplets_dir.split('/')
    split = dir_list[-1] if dir_list[-1] else dir_list[-2]
    results_path = os.path.join(out_path, f'{p:02d}', f'{rnd_seed:02d}', split)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    _save_weights(trainer=nmf_trainer, results_path=results_path, format=out_format)

def _save_weights(trainer:object, results_path:str, format:str) -> None:
    W_nmf = trainer.nmf.W.weight.data.detach().abs().numpy()
    if format == 'txt':
        np.savetxt(os.path.join(results_path, 'nmf_components.txt'), W_nmf)
    elif format == 'npy':
        with open(os.path.join(results_path, 'nmf_components.npy'), 'wb') as f:
            np.save(f, W_nmf)
    else:
        scipy.io.savemat(os.path.join(results_path, 'nmf_components.mat'), {'components': W_nmf})

if __name__ == '__main__':
    for rnd_seed in range(0, 20):
        torch.multiprocessing.set_start_method('spawn', force=True)
        args = parseargs()
        np.random.seed(rnd_seed)
        random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        #np.random.seed(args.rnd_seed)
        #random.seed(args.rnd_seed)
        #torch.manual_seed(args.rnd_seed)
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
            args.alpha,
            args.optimizer,
            args.criterion,
            args.n_components,
            args.out_format,
            args.batch_size,
            args.epochs,
            rnd_seed, #args.rnd_seed
            device,
            args.init_weights,
            args.window_size,
            args.verbose,
            ),
            nprocs=n_subprocs,
            join=True)
