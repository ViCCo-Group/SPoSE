#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import json
import os
import re
import shutil
import sys

import numpy as np
from typing import List, Tuple


def del_paths(results: dict, best_lmbda: float) -> None:
    for lmbda, values in results.items():
        if lmbda != best_lmbda:
            for root in values['roots']:
                shutil.rmtree(root)
                try:
                    dir_list = root.split('/')
                    dir_list[1] = 'plots'
                    plots_path = '/'.join(dir_list)
                    shutil.rmtree(plots_path)
                except FileNotFoundError:
                    continue


def keep_final_models(PATH: str) -> None:
    models = sorted([os.path.join(root, name) for root, _, files in os.walk(
        PATH) for name in files if name.endswith('.tar')])
    weights = sorted([os.path.join(root, name) for root, _, files in os.walk(
        PATH) for name in files if name.endswith('.txt')])
    _ = models.pop()
    _ = weights.pop()
    for model, weight in zip(models, weights):
        os.remove(model)
        os.remove(weight)


def crawl(PATH: str, dim: int) -> dict:
    results = defaultdict(lambda: defaultdict(dict))
    for split in os.scandir(PATH):
        print(f'Currently crawling directories for {split.name}\n')
        if split.is_dir() and re.search(r'(?=.*split)(?=.*\d+$)', split.name): 
            i = 0
            for lmbda in os.scandir(os.path.join(PATH, split.name, f'{dim}d')):
                #if seed.is_dir() and re.search(r'(?=^seed)(?=.*\d+$)', seed.name):
                if lmbda.is_dir() and re.search(r'\d+', lmbda.name):
                    for root, _, files in os.walk(os.path.join(PATH, split.name, f'{dim}d', lmbda.name)):
                        if files:
                            files = sorted([f for f in files if re.search(r'(?=^results)(?=.*json$)', f)])
                            f = files[-1]
                            seed = root.split('/')[-1]
                            with open(os.path.join(root, f), 'r') as f:
                                val_loss = json.load(f)['val_loss']
                                if 'roots' in results[split.name][lmbda.name]:
                                    results[split.name][lmbda.name]['roots'].append(root)
                                    results[split.name][lmbda.name]['seeds'].append(seed)
                                else:
                                    results[split.name][lmbda.name]['roots'] = [root]
                                    results[split.name][lmbda.name]['seeds'] = [seed]
                                if np.isnan(val_loss):
                                    print(
                                        f'Found NaN in cross-entropy loss for: {lmbda.name}')
                                    val_loss = np.inf
                                if 'cross-entropies' in results[split.name][lmbda.name]:
                                    results[split.name][lmbda.name]['cross-entropies'].append(val_loss)
                                else:
                                    results[split.name][lmbda.name]['cross-entropies'] = [val_loss]
                    i += 1
                else:
                    print(
                        f'{os.path.join(PATH, split.name, seed)} does not seem to be a valid directory.\n')
            if not i:
                raise Exception(
                    'Crawling the provided path for results was not successful. Make sure to provide a path containing results for all seeds.\n')
    return results


def find_best_lmbda(split: dict, dim: int):
    mean_centropies = {lmbda: np.mean(values['cross-entropies']) for lmbda, values in split.items()}
    best_lmbda = min(mean_centropies.items(), key=lambda kv:kv[1])[0]
    del_paths(split, best_lmbda)
    return best_lmbda


def get_model_links(split, best_lmbda, dim):
    best_hyper = split[best_lmbda]
    links = ['/'.join((f'{dim}d', seed, best_lmbda, 'model', 'model_epoch2000.tar')) for seed in best_hyper['seeds']]
    return links


def find_best_hypers(PATH: str, results: dict, dim: int):
    with open(os.path.join(PATH, 'model_links.txt'), 'w') as f:
        for split, values in results.items():
            best_lmbda = find_best_lmbda(values, dim)
            links = get_model_links(values, best_lmbda, dim)
            for link in links:
                f.write('/'.join((PATH, split, link)))
                #f.write(f'/'.join((PATH.split('/')[-1], split, link)))
                f.write('\n')
            for root in values[best_lmbda]['roots']:
                try:
                    keep_final_models(root)
                except:
                    continue


if __name__ == '__main__':
    PATH = sys.argv[1]
    dim = sys.argv[2]
    results = crawl(PATH, dim)
    find_best_hypers(PATH, results, dim)

