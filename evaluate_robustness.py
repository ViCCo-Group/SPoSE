#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import random
import re
import torch
import utils
import itertools

import numpy as np

from models.model import SPoSE
from typing import Tuple, List, Dict

os.environ['PYTHONIOENCODING']='UTF-8'
os.environ['OMP_NUM_THREADS']='1' #number of cores used per Python process (set to 2 if HT is enabled, else keep 1)

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='path/to/triplets/data')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_items', type=int,
        help='number of unique items in dataset')
    aa('--dim', type=int, default=100,
        help='latent dimensionality of VSPoSE embedding matrices')
    aa('--thresh', type=float, default=0.8,
        choices=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help='examine fraction of dimensions across models that is above threshold (corresponds to Pearson correlation)')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--device', type=str,
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def avg_ndims(Ws_mu:list) -> np.ndarray:
    return np.ceil(np.mean(list(map(lambda w: min(w.shape), Ws_mu))))

def std_ndims(Ws_mu:list) -> np.ndarray:
    return np.std(list(map(lambda w: min(w.shape), Ws_mu)))

def robustness(corrs:np.ndarray, thresh:float) -> float:
    return len(corrs[corrs>thresh])/len(corrs)

def compare_dimensions(Ws_mu:list, thresh:float) -> Tuple[np.ndarray]:
    N = max(Ws_mu[0].shape)
    rnd_perm = np.random.permutation(N)
    train_indices = rnd_perm[:int(N*.8)]
    test_indices = rnd_perm[int(N*.8):]
    loc_robustness_scores = []
    scale_robustness_scores = []
    for i, W_mu_i in enumerate(Ws_mu):
        for j, W_mu_j in enumerate(Ws_mu):
            if i != j:
                assert max(W_mu_i.shape) == max(W_mu_j.shape), '\nNumber of items in weight matrices must align.\n'
                corrs = np.zeros(min(W_mu_i.shape))
                W_mu_i_train, W_mu_j_train = W_mu_i[:, train_indices], W_mu_j[:, train_indices]
                W_mu_i_test, W_mu_j_test = W_mu_i[:, test_indices], W_mu_j[:, test_indices]
                for k, w_i in enumerate(W_mu_i_train):
                    argmax = np.argmax([utils.pearsonr(w_i, w_j) for w_j in W_mu_j_train])
                    corrs[k] = utils.pearsonr(W_mu_i_test[k], W_mu_j_test[argmax])
                rel_freq = robustness(corrs, thresh)
                loc_robustness_scores.append(rel_freq)
    avg_loc_robustness = np.mean(loc_robustness_scores)
    std_loc_robustness = np.std(loc_robustness_scores)
    avg_scale_robustness = np.mean(scale_robustness_scores)
    return avg_loc_robustness, std_loc_robustness, avg_scale_robustness

def estimate_redundancy_(Ws_mu:list) -> Tuple[float, float]:
    def cosine_sim(x:np.ndarray, y:np.ndarray) -> float:
        return (x @ y) / (np.linalg.norm(x)*np.linalg.norm(y))
    def get_redundant_pairs(W:np.ndarray, thresh:float=.9) -> int:
        w_combs = list(itertools.combinations(W, 2))
        cosine_sims = np.array([cosine_sim(w_i, w_j) for (w_i, w_j) in w_combs])
        n_redundant_pairs = np.where(cosine_sims > thresh, 1, 0).sum()
        return n_redundant_pairs
    def get_redundant_dims(W:np.ndarray, thresh:float=.9) -> int:
        n_redundant_dims = 0
        for i, w_i in enumerate(W):
            for j, w_j in enumerate(W):
                if i != j:
                    cos_sim = cosine_sim(w_i, w_j)
                    if cos_sim > thresh:
                        n_redundant_dims += 1
                        print(f'\nFound redundant dimension with cross-cosine similarity: {cos_sim.round(3)}.\n')
                        break
        return n_redundant_dims
    avg_redundant_pairs = np.mean(list(map(get_redundant_pairs, Ws_mu)))
    avg_redundant_dims = np.mean(list(map(get_redundant_dims, Ws_mu)))
    return avg_redundant_pairs, avg_redundant_dims

def compute_robustness(Ws_mu:list, thresh:float=.9):
    avg_loc_robustness, std_loc_robustness, avg_scale_robustness = compare_dimensions(Ws_mu, thresh)
    scores = {}
    scores['avg_loc_robustness'] = avg_loc_robustness
    scores['std_loc_robustness'] = std_loc_robustness
    scores['avg_scale_robustness'] = avg_scale_robustness
    scores['avg_sparsity'] = utils.avg_sparsity(Ws_mu)
    scores['avg_ndims'] = avg_ndims(Ws_mu)
    scores['std_ndims'] = std_ndims(Ws_mu)
    scores['hist'] = list(map(lambda W: W.shape[0], Ws_mu))
    n_redundant_pairs, n_redundant_dims = estimate_redundancy_(Ws_mu)
    scores['n_redundant_pairs'] = n_redundant_pairs
    scores['n_redundant_dims'] = n_redundant_dims
    return scores

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

def evaluate_models(
                    results_dir:str,
                    triplets_dir:str,
                    n_items:int,
                    dim:int,
                    thresh:float,
                    device:torch.device,
                    task=None,
                    batch_size=None,
                    ) -> None:
    PATH = os.path.join(results_dir, f'{dim}d')
    model_paths = get_model_paths(PATH)
    Ws_mu = []
    
    _, val_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir, inference=False)
    val_batches = utils.load_batches(
        train_triplets=None, test_triplets=val_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    val_losses = np.zeros(len(model_paths))
    for i, model_path in enumerate(model_paths):
        model = SPoSE(in_size=n_items, out_size=dim, init_weights=True)
        model.to(device)
        try:
            model = utils.load_model(model, model_path, device)
            W = model.fc.weight.data.cpu().numpy()
            W_mu = utils.remove_zeros(np.copy(W))
        except FileNotFoundError:
            raise Exception(f'Could not find final weights for {model_path}\n')
        _, _, val_loss, _, _ = utils.test(W=W_mu, test_batches=val_batches, task=task, device=device, batch_size=batch_size)
        val_losses[i] += val_loss
        Ws_mu.append(W_mu)

    model_robustness = compute_robustness(Ws_mu, thresh=thresh)
    print(f"\nRobustness scores for latent dim = {dim}: {model_robustness}\n")

    out_path = os.path.join(PATH, 'robustness_scores', str(thresh))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, 'robustness.txt'), 'wb') as f:
        f.write(pickle.dumps(model_robustness))

    with open(os.path.join(PATH, 'val_entropies.npy'), 'wb') as f:
        np.save(f, val_losses)

if __name__ == '__main__':
    args = parseargs()
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    evaluate_models(
                    results_dir=args.results_dir,
                    triplets_dir=args.triplets_dir,
                    task=args.task,
                    n_items=args.n_items,
                    dim=args.dim,
                    thresh=args.thresh,
                    device=args.device,
                    batch_size=args.batch_size,
                    )