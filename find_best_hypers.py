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
        plots_path = path.split('/')
        plots_path[1] = 'plots'
        plots_path = '/'.join(plots_path)
        shutil.rmtree(plots_path)

def find_best_hypers_(PATH:str) -> Tuple[str, float]:
    paths, results = [], []
    for root, _, files in os.walk(PATH):
        for name in files:
            if name.endswith('.json'):
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

if __name__ == '__main__':
    PATH = sys.argv[1]
    find_best_hypers_(PATH)
