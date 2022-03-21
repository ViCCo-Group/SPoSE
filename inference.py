#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from email.policy import default
import os
import re
import torch
import utils

import numpy as np

from collections import defaultdict
from models.model import SPoSE
from typing import List, Dict

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_items', type=int, default=1854,
        help='number of unique items/objects in dataset')
    aa('--dim', type=int, default=100,
        help='latent dimensionality of SPoSE embedding matrices')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='directory from where to load test triplets data')
    args = parser.parse_args()
    return args

def get_model_paths(PATH:str) -> List[str]:
    model_paths = []
    regex = '(?=^model)(?=.*\d)(?=.*tar$)'
    for seed in os.scandir(PATH):
        if seed.is_dir() and seed.name[-2:].isdigit():
            seed_path = os.path.join(PATH, seed.name)
            for root, _, files in os.walk(seed_path):
                files = [f for f in files if re.compile(regex).search(f)]
                if files:
                    model_paths.append(root)
    return model_paths

def entropy_(p:np.ndarray) -> np.ndarray:
    return np.sum(np.where(p == 0, 0, p*np.log(p)))

def cross_entropy_(p:np.ndarray, q:np.ndarray) -> float:
    return -np.sum(p*np.log(q))

def kld_(p:np.ndarray, q:np.ndarray) -> float:
    return entropy_(p) + cross_entropy_(p, q)

def l1_distance(p:np.ndarray, q:np.ndarray) -> float:
    return np.linalg.norm(p - q, ord=1)

def compute_divergences(human_pmfs:dict, model_pmfs:dict, metric:str='kld'):
    assert len(human_pmfs) == len(model_pmfs), '\nNumber of triplets in human and model distributions must correspond.\n'
    divergences = np.zeros(len(model_pmfs))
    accuracy = 0
    for i, (triplet, p) in enumerate(human_pmfs.items()):
        q = np.asarray(model_pmfs[triplet])
        if metric  == 'kld':
            div = kld_(p, q)
        elif metric == 'cross-entropy':
            div = cross_entropy_(p, q)
        else:
            div = l1_distance(p, q)
        divergences[i] += div
    return divergences

def inference(
             task:str,
             n_items:int,
             dim:int,
             batch_size:int,
             results_dir:str,
             triplets_dir:str,
             device:torch.device,
             ) -> None:
    PATH = os.path.join(results_dir, f'{dim}d')
    model_paths = get_model_paths(PATH)
    test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir,  inference=True)
    _, val_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=False)

    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    val_batches = utils.load_batches(train_triplets=None, test_triplets=val_triplets, n_items=n_items, batch_size=batch_size, inference=True)

    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    test_accs = dict()
    test_losses = dict()
    val_losses = dict()
    model_pmfs_all = defaultdict(dict)
    model_choices = defaultdict(list)

    for model_path in model_paths:
        seed = model_path.split('/')[-2]
        model =  SPoSE(in_size=n_items, out_size=dim, init_weights=True)

        try:
            model = utils.load_model(model, model_path, device)
            W = model.fc.weight.data.cpu().numpy()
        except FileNotFoundError:
            raise Exception(f'\nCannot find weight matrices in: {model_path}\n')

        W_pruned = utils.remove_zeros(np.copy(W))
        if not min(W_pruned.shape):
            W_pruned = np.copy(W)
        W = W_pruned

        triplet_choices, test_acc, test_loss, probas, model_pmfs = utils.test(W=W, test_batches=test_batches, task=task, device=device, batch_size=batch_size)
        _, _, val_loss, _, _ = utils.test(W=W, test_batches=val_batches, task=task, device=device, batch_size=batch_size)

        print(f'\nTest accuracy for current random seed: {test_acc}')
        print(f'Validation cross-entropy error for current random seed: {val_loss}\n')

        test_accs[seed] = test_acc
        val_losses[seed] = val_loss
        test_losses[seed] = test_loss
        model_pmfs_all[seed] = model_pmfs
        model_choices[seed].extend(triplet_choices)

        with open(os.path.join(model_path, 'test_probas.npy'), 'wb') as f:
            np.save(f, probas)

    avg_test_acc = np.mean(list(test_accs.values()))
    median_test_acc = np.median(list(test_accs.values()))
    max_test_acc = max(list(test_accs.values()))
    print(f'\nMean accuracy on held-out test set: {avg_test_acc}')
    print(f'Median accuracy on held-out test set: {median_test_acc}')
    print(f'Max accuracy on held-out test set: {max_test_acc}\n')

    PATH = os.path.join(PATH, 'evaluation_metrics')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    test_accs = dict(sorted(test_accs.items(), key=lambda kv:kv[1], reverse=False))
    test_losses = dict(sorted(test_losses.items(), key=lambda kv:kv[1], reverse=True))
    val_losses = dict(sorted(val_losses.items(), key=lambda kv:kv[1], reverse=True))
    #NOTE: we leverage the model that is slightly better than the median model (since we have 20 random seeds, the median is the average between model 10 and 11)
    median_model = list(val_losses.keys())[int(len(val_losses)//2)]
    median_model_pmfs = model_pmfs_all[median_model]
    best_model_pmfs = model_pmfs_all[list(val_losses.keys())[-1]]
    median_model_choices = model_choices[median_model]

    utils.pickle_file(test_accs, PATH, 'test_accuracies')
    utils.pickle_file(test_losses, PATH, 'test_losses')
    utils.pickle_file(median_model_pmfs, PATH, 'median_model_choice_pmfs')
    utils.pickle_file(best_model_pmfs, PATH, 'model_choice_pmfs_best')
    utils.pickle_file(median_model_choices, PATH, 'triplet_choices')

    human_pmfs = utils.unpickle_file(triplets_dir, 'human_choice_pmfs')

    klds_median = compute_divergences(human_pmfs, median_model_pmfs, metric='kld')
    klds_best = compute_divergences(human_pmfs, best_model_pmfs, metric='kld')

    cross_entropies = compute_divergences(human_pmfs, median_model_pmfs, metric='cross-entropy')
    l1_distances = compute_divergences(human_pmfs, median_model_pmfs, metric='l1-distance')

    np.savetxt(os.path.join(PATH, 'klds_median.txt'), klds_median)
    np.savetxt(os.path.join(PATH, 'klds_best.txt'), klds_best)

    np.savetxt(os.path.join(PATH, 'cross_entropies_median.txt'), cross_entropies)
    np.savetxt(os.path.join(PATH, 'l1_distances_median.txt'), l1_distances)

    klds, cross_entropies, l1_distances = {}, {}, {}
    for seed, model_pmfs in model_pmfs_all.items():
        klds[seed] = np.mean(compute_divergences(human_pmfs, model_pmfs, metric='kld'))
        cross_entropies[seed] = np.mean(compute_divergences(human_pmfs, model_pmfs, metric='cross-entropy'))
        l1_distances[seed] = np.mean(compute_divergences(human_pmfs, model_pmfs, metric='l1-distance'))

    utils.pickle_file(klds, PATH, 'klds_all')
    utils.pickle_file(cross_entropies, PATH, 'cross_entropies_all')
    utils.pickle_file(l1_distances, PATH, 'l1_distances_all')

if __name__ == '__main__':
    args = parseargs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference(
              task=args.task,
              n_items=args.n_items,
              dim=args.dim,
              batch_size=args.batch_size,
              results_dir=args.results_dir,
              triplets_dir=args.triplets_dir,
              device=device,
              )
