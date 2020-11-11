#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models.model import *

os.environ['PYTHONIOENCODING']='UTF-8'

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--version', type=str, default='deterministic',
        choices=['deterministic', 'variational'],
        help='whether to apply a deterministic or variational version of SPoSE')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--modality', type=str, default='behavioral/',
        #choices=['behavioral/', 'text/', 'visual/', 'neural/'],
        help='define for which modality SPoSE should be perform specified task')
    aa('--triplets_dir', type=str, default='./triplets',
        help='in case you have tripletized data, provide directory from where to load triplets')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/modality/version/dim/lambda/rnd_seed/)')
    aa('--embed_dim', metavar='D', type=int, default=90,
        help='dimensionality of the embedding matrix (i.e., out_size of model)')
    aa('--lmbda', type=float,
        help='lambda value determines scale of l1 regularization')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def run(
        n_samples:int,
        version:str,
        task:str,
        modality:str,
        triplets_dir:str,
        lmbda:float,
        batch_size:int,
        embed_dim:int,
        rnd_seed:int,
        device:torch.device,
) -> None:
    #load train triplets
    train_triplets, _ = load_data(device=device, triplets_dir=os.path.join(triplets_dir, modality))
    #number of unique items in the data matrix
    n_items = torch.max(train_triplets).item() + 1
    #initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    #get mini-batches for training to sample an equally sized synthetic dataset
    train_batches = BatchGenerator(I=I, dataset=train_triplets, batch_size=batch_size, sampling_method=None, p=None)
    #initialise model
    for i in range(n_samples):
        if version == 'variational':
            model = VSPoSE(in_size=n_items, out_size=embed_dim)
        else:
            model = SPoSE(in_size=n_items, out_size=embed_dim)
        #load weights of pretrained model
        model = load_model(
                           model=model,
                           results_dir=results_dir,
                           modality=modality,
                           version=version,
                           dim=embed_dim,
                           lmbda=lmbda,
                           rnd_seed=rnd_seed,
                           device=device,
                           )
        #move model to current device
        model.to(device)
        #probabilistically sample triplet choices given the model's ouput PMFs
        sampled_choices = validation(
                                    model=model,
                                    val_batches=train_batches,
                                    version=version,
                                    task=task,
                                    device=device,
                                    embed_dim=embed_dim,
                                    sampling=True,
                                    )

        np.savetxt(f'./triplets/synthetic/sample_{i:02d}/train_90.txt', sampled_choices)

if __name__ == '__main__':
    #parse all arguments and set random seeds
    args = parseargs()

    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    #set device
    device = torch.device(args.device)

    #some variables to debug / potentially resolve CUDA problems
    if device == torch.device('cuda:0'):
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)

    elif device == torch.device('cuda:1') or device == torch.device('cuda'):
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(1)

    print(f'PyTorch CUDA version: {torch.version.cuda}')
    print()

    run(
        n_samples=args.n_samples,
        version=args.version,
        task=args.task,
        modality=args.modality,
        triplets_dir=args.triplets_dir,
        lmbda=args.lmbda,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        rnd_seed=args.rnd_seed,
        device=args.device,
        )
