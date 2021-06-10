#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import sys

import numpy as np
from typing import List, Tuple

def del_paths_(paths:List[str]) -> None:
    for path in paths:
        shutil.rmtree(path)
        try:
            dir_list = path.split('/')
            dir_list[1] = 'plots'
            plots_path = '/'.join(dir_list)
            shutil.rmtree(plots_path)
        except FileNotFoundError:
            pass

def keep_final_epoch_(PATH:str) -> None:
    models = sorted([os.path.join(root, name) for root, _, files in os.walk(PATH) for name in files if name.endswith('.tar')])
    weights = sorted([os.path.join(root, name) for root, _, files in os.walk(PATH) for name in files if name.endswith('.txt')])
    _ = models.pop()
    _ = weights.pop()
    for model, weight in zip(models, weights):
        os.remove(model)
        os.remove(weight)

def find_best_hypers_(PATH:str) -> Tuple[str, float]:
    paths, results = [], []
    for root, _, files in os.walk(PATH):
        for name in files:
            if name == 'results_0500.json':
            #if name.endswith('.json'):
                paths.append(root)
                with open(os.path.join(root, name), 'r') as f:
                    val_loss = json.load(f)['val_loss']
                    if np.isnan(val_loss):
                        print(f'Found NaN in cross-entropy loss for: {root}')
                        results.append(np.inf)
                        continue
                    results.append(val_loss)
    if sum(np.isinf(results)) == len(results):
        raise Exception(f'Found NaN values in cross-entropy loss for every model. Change lambda value grid.')
    argmin_loss = np.argmin(results)
    best_model = paths.pop(argmin_loss)
    print(f'Best params: {best_model}\n')
    del_paths_(paths)
    keep_final_epoch_(best_model)

if __name__ == '__main__':
    PATH = sys.argv[1]
    i = 0
    for d in os.scandir(PATH):
        if d.is_dir() and d.name.startswith('seed'):
            find_best_hypers_(os.path.join(PATH, d.name))
            i += 1
        else:
            print(f'{os.path.join(PATH, d.name)} does not seem to be a valid directory.\n')
    if not i:
        raise Exception('Crawling the provided path for results was not successful. Make sure to provide a path containing results for all seeds.\n')
