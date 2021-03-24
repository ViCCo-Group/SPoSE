#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import torch
import utils

import numpy as np

from collections import defaultdict
from models.model import SPoSE

def inference(
             modality:str,
             task:str,
             dim:int,
             batch_size:int,
             results_dir:str,
             triplets_dir:str,
             device:torch.device,
             ) -> None:

    PATH = os.path.join(results_dir, modality, 'deterministic', f'{dim}d')
    model_paths = [root for root, _, files in os.walk(PATH) for name in files if name.endswith('.json')]

    test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=True)
    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=1854, batch_size=batch_size, inference=True)
    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    test_accs = dict()
    model_pmfs_all = defaultdict(dict)

    for model_path in model_paths:
        try:
            W = utils.load_weights(model_path)
        except FileNotFoundError:
            raise Exception(f'\nCannot find weight matrices in: {model_path}\n')

        test_acc, probas, model_pmfs = utils.test(W=W, test_batches=test_batches, task=task, device=device, batch_size=batch_size)

        seed = model_path.split('/')[-3]
        test_accs[seed] = test_acc
        model_pmfs_all[seed] = model_pmfs

        with open(os.path.join(model_path, 'test_probas.npy'), 'wb') as f:
            np.save(f, probas)

    avg_test_acc = np.mean(test_accs)
    print(f'\nMean accuracy on held-out test set: {avg_test_acc}\n')

    PATH = os.path.join(model_path, 'evaluation_metrics')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    utils.pickle_file(model_pmfs_all, PATH, 'model_choice_pmfs')

    with open(os.path.join(PATH, 'test_accuracies.npy'), 'wb') as f:
        np.save(f, test_accs)

if __name__ == '__main__':
    #parse os argument variables
    modality = sys.argv[1]
    task = sys.argv[3]
    dim = int(sys.argv[4])
    batch_size = int(sys.argv[6])
    results_dir = sys.argv[7]
    triplets_dir = sys.argv[8]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference(
              modality=modality,
              task=task,
              dim=dim,
              batch_size=batch_size,
              results_dir=results_dir,
              triplets_dir=triplets_dir,
              device=device,
              )
