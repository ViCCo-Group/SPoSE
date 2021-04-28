#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
import torch
import utils

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from models.model import SPoSE
from typing import List, Dict

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--modality', type=str,
        help='current modality (e.g., behavioral, synthetic)')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_items', type=int, default=1854,
        help='number of unique items/objects in dataset')
    aa('--dim', type=int, default=100,
        help='latent dimensionality of VSPoSE embedding matrices')
    aa('--e_temps', type=float, nargs='+',
        help='temperature values for scaling the embeddings')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets data')
    aa('--device', type=str,
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'])
    args = parser.parse_args()
    return args

def get_model_paths(PATH:str) -> List[str]:
    model_paths = []
    for seed in os.scandir(PATH):
        if seed.is_dir() and seed.name[-2:].isdigit():
            seed_path = os.path.join(PATH, seed.name)
            for root, dirs, files in os.walk(seed_path):
                 for name in files:
                      if name.endswith('.json'):
                          model_paths.append(root)
    return model_paths

def inference(
             process_id:int,
             modality:str,
             task:str,
             n_items:int,
             dim:int,
             e_temps:List[float],
             batch_size:int,
             results_dir:str,
             triplets_dir:str,
             device:torch.device,
             ) -> None:

    PATH = os.path.join(results_dir, modality, 'deterministic', f'{dim}d')
    model_paths = get_model_paths(PATH)
    e_temp = e_temps[process_id]
    _, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=False)
    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    val_centropies = dict()
    val_accs = dict()

    for model_path in model_paths:
        seed = model_path.split('/')[-2]
        try:
            W = np.loadtxt(os.path.join(model_path, utils.load_weights(model_path)))
        except FileNotFoundError:
            raise Exception(f'\nCannot find weight matrices in: {model_path}\n')

        val_acc, val_loss, probas, _ = utils.test(W=W, test_batches=test_batches, task=task, batch_size=batch_size, device=device, e_temp=e_temp)
        val_centropies[seed] = val_loss
        val_accs[seed] = val_acc

        print(f'Validation accuracy for current random seed: {val_acc}')
        print(f'Validation cross-entropy for current random seed: {val_loss}\n')

        f_name = 'val_probas.npy'
        with open(os.path.join(model_path, f_name), 'wb') as f:
            np.save(f, probas)

    val_accs_ = list(val_accs.values())
    avg_val_acc = np.mean(val_accs_)
    median_val_acc = np.median(val_accs_)
    max_val_acc = np.max(val_accs_)
    print(f'\nMean accuracy on validation set: {avg_val_acc}')
    print(f'Median accuracy on validation set: {median_val_acc}')
    print(f'Max accuracy on validation set: {max_val_acc}\n')

    PATH = os.path.join(PATH, 'evaluation_metrics', 'validation', f'{e_temp:.2f}')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    utils.pickle_file(val_accs, PATH, 'val_accs')
    utils.pickle_file(val_centropies, PATH, 'val_centropies')

if __name__ == '__main__':
    args = parseargs()
    n_procs = len(args.e_temps)
    torch.multiprocessing.set_start_method('spawn', force=True)

    if re.search(r'^cuda', args.device):
        try:
            current_device = int(args.device[-1])
        except ValueError:
            current_device = 1
        try:
            torch.cuda.set_device(current_device)
        except RuntimeError:
            torch.cuda.set_device(0)
        print(f'\nPyTorch CUDA version: {torch.version.cuda}')
    else:
        if n_procs > os.cpu_count()-1:
            raise Exception(f'CPU node cannot run {n_procs} in parallel. Maximum number of processes is {os.cpu_count()-1}.\n')

    print(f'\nRunning {n_procs} processes in parallel.\n')
    torch.multiprocessing.spawn(
                                inference,
                                args=(
                                args.modality,
                                args.task,
                                args.n_items,
                                args.dim,
                                args.e_temps,
                                args.batch_size,
                                args.results_dir,
                                args.triplets_dir,
                                args.device,
                                ),
                                nprocs=n_procs,
                                join=True)
