#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import torch
import scipy.io
import numpy as np

os.environ['PYTHONIOENCODING']='UTF-8'

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--in_path', type=str,
        help='folder where to find word embeddings or image features')
    aa('--out_path', type=str,
        help='folder where to store triplets')
    aa('--method', type=str,
        choices=['deterministic', 'probabilistic'],
        help='whether to deterministically (argmax) or probabilistically (conditioned on PMF) sample odd-one-out choices')
    aa('--temperature', type=float, default=None,
        help='softmax temperature (beta param) in probabilistic tripletizing approach')
    aa('--n_samples', type=float,
        help='number of triplet samples')
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def load_data(in_path:str) -> np.ndarray:
    if re.search(r'(mat|txt|csv|npy)$', in_path):
        try:
            if re.search(r'mat$', in_path):
                X = np.vstack([v for v in scipy.io.loadmat(in_path).values() if isinstance(v, np.ndarray) and v.dtype == np.float])
            elif re.search(r'txt$', in_path):
                X = np.loadtxt(in_path)
            elif re.search(r'csv$', in_path):
                X = np.loadtxt(in_path, delimiter=',')
            else:
                with open(in_path, 'rb') as f:
                    X = np.load(f)
            X = remove_nans_(X)
            return X
        except:
            raise Exception('\nInput data is not in the correct format\n')
    else:
        raise Exception('\nCannot tripletize input data other than .mat, .txt, .csv, .npy\n')

def remove_nans_(X:np.ndarray) -> np.ndarray:
    nan_indices = np.isnan(X[:, :]).any(axis=1)
    return X[~nan_indices]

def filter_triplets(rnd_samples:np.ndarray, n_samples:float) -> np.ndarray:
    """filter for unique triplets (i, j, k have to be different indices)"""
    def is_set_(triplet:np.ndarray) -> bool:
        return len(np.unique(triplet)) == len(triplet)
    rnd_samples = np.asarray(list(filter(is_set_, rnd_samples)))
    rnd_samples = np.unique(rnd_samples, axis=0)[:int(n_samples)]
    return rnd_samples

def tripletize_(
                in_path:str,
                out_path:str,
                method:str,
                temperature:float,
                n_samples:float,
) -> None:
    """create triplets of object embedding similarities, and for each triplet find the odd-one-out"""
    sampling_constant = n_samples / 10
    #load input data (e.g., word embeddings, image features)
    X = load_data(in_path)
    #create similarity matrix
    #TODO: figure out whether an affinity matrix might be more reasonable (i.e., informative) than a simple similarity matrix
    S = X @ X.T
    N = S.shape[0]
    #draw random triplet samples
    rnd_samples = np.random.randint(N, size=(int(n_samples + sampling_constant), 3))
    #filter for unique triplets and remove all duplicates
    rnd_samples = filter_triplets(rnd_samples, n_samples)

    if method == 'probabilistic':
        assert isinstance(temperature, float), '\nFloat for softmax temperature is required in probabilistic approach\n'
        max_probas = np.zeros(int(n_samples))
        def softmax(x:np.ndarray, temperature:float) -> np.ndarray:
            return np.exp(temperature * x)/np.sum(np.exp(temperature * x))

        def sample_choices(odd_one_outs:np.ndarray, sims:np.ndarray, temperature:float) -> np.ndarray:
            """sample triplet choices probabilistically (conditioned on PMF obtained by softmax over similarity values)"""
            probas = softmax(sims, temperature)
            choices = np.random.choice(odd_one_outs, size=len(probas), replace=False, p=probas)
            choices = choices[::-1]
            return choices, max(probas)

    triplets = np.zeros((int(n_samples), 3), dtype=int)
    for idx, [i, j, k] in enumerate(rnd_samples):
        odd_one_outs = np.asarray([k, j, i])
        sims = np.array([S[i, j], S[i, k], S[j, k]])
        if method == 'probabilistic':
            choices, max_p = sample_choices(odd_one_outs, sims, temperature)
            max_probas[idx] += max_p
        else:
            #simply use the argmax to (deterministically) find the odd-one-out choice
            choices = odd_one_outs[np.argsort(sims)]
        triplets[idx] = choices

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    rnd_indices = np.random.permutation(len(triplets))
    train_triplets = triplets[rnd_indices[:int(len(rnd_indices)*.9)]]
    test_triplets = triplets[rnd_indices[int(len(rnd_indices)*.9):]]

    with open(os.path.join(out_path, 'train_90.npy'), 'wb') as train_file:
        np.save(train_file, train_triplets)

    with open(os.path.join(out_path, 'test_10.npy'), 'wb') as test_file:
        np.save(test_file, test_triplets)

if __name__ == "__main__":
    #parse all arguments
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    #tripletize data
    tripletize_(
                in_path=args.in_path,
                out_path=args.out_path,
                method=args.method,
                temperature=args.temperature,
                n_samples=args.n_samples,
    )
