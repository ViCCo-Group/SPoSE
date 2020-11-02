#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'plot_nneg_dims_over_time',
            'plot_multiple_performances',
            'plot_single_performance',
            ]

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_nneg_dims_over_time(
                            nneg_d_over_time:list,
                            lmbda:float,
                            folder:str,
                            version:str,
) -> None:
    """plot number of non-negative dimensions as a function of time (i.e., epochs)"""
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    epochs, nneg_dims = zip(*nneg_d_over_time)
    ax.plot(epochs, nneg_dims, '-o')
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Number of non-negative dimensions')

    dir = os.path.join('./plots/', folder, version, 'nneg_dimensions/')
    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.tight_layout()
    plt.savefig(dir + 'nneg_dimensions_over_time' + '_' + str(lmbda) + '.png')
    plt.close()

def plot_single_performance(
                            val_accs:list,
                            train_accs:list,
                            lmbda:float,
                            folder:str,
                            version:str,
) -> None:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.plot(val_accs,'-+',  alpha=.5, label='Test')
    ax.plot(train_accs, '-+', alpha=.5, label='Train')
    ax.annotate('Val acc: {:.3f}'.format(np.max(val_accs)), (len(val_accs) - len(val_accs) * 0.1, np.max(val_accs) / 2))
    ax.set_xlim([0, len(val_accs)])
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'Accuracy')
    ax.set_title(f'Lambda-L1: {lmbda}')
    ax.legend(fancybox=True, shadow=True, loc='lower left')

    dir = os.path.join('./plots/', folder, version, 'grid_search/')
    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.tight_layout()
    plt.savefig(dir + 'single_model_performance_over_time' + '_' + str(lmbda) + '.png')
    plt.close()

def plot_multiple_performances(
                                val_accs:list,
                                train_accs:list,
                                lambdas:np.ndarray,
                                folder:str,
                                version:str,
) -> None:
    n_rows = len(lambdas) // 2
    n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=100)
    max_conv = max(list(map(lambda accs: len(accs), val_accs)))

    #keep track of k
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            #hide the right and top spines
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['top'].set_visible(False)

            #only show ticks on the left (y-axis) and bottom (x-axis) spines
            axes[i, j].yaxis.set_ticks_position('left')
            axes[i, j].xaxis.set_ticks_position('bottom')

            axes[i, j].plot(val_accs[k],'-+',  alpha=.5, label='Test')
            axes[i, j].plot(train_accs[k], '-+', alpha=.5, label='Train')
            axes[i, j].annotate('Val acc: {:.3f}'.format(np.max(val_accs)), (max_conv - max_conv * 0.1, np.max(val_accs) / 2))
            axes[i, j].set_xlim([0, max_conv])
            axes[i, j].set_xlabel(r'Epochs')
            axes[i, j].set_ylabel(r'Accuracy')
            axes[i, j].set_title(f'Lambda-L1: {lambdas[k]}')
            axes[i, j].legend(fancybox=True, shadow=True, loc='lower left')
            k += 1

    for ax in axes.flat:
        ax.label_outer()

    dir = os.path.join('./plots/', folder, version, 'grid_search/')
    if not os.path.exists(dir):
        os.makedirs(dir)

    fig.tight_layout()
    plt.savefig(dir + 'model_performances_over_time.png')
    plt.clf()
    plt.close()
