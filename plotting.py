#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'plot_aggregated_klds',
            'plot_grid_search_results',
            'plot_kld_violins',
            'plot_nneg_dims_over_time',
            'plot_multiple_performances',
            'plot_pruning_results',
            'plot_single_performance',
            ]
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_nneg_dims_over_time(plots_dir:str, nneg_d_over_time:list) -> None:
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

    PATH = os.path.join(plots_dir, 'nneg_dimensions')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, 'nneg_dimensions_over_time.png'))
    plt.close()

def plot_single_performance(plots_dir:str, val_accs:list, train_accs:list) -> None:
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

    PATH = os.path.join(plots_dir, 'grid_search')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, 'single_model_performance_over_time.png'))
    plt.close()

def plot_multiple_performances(
                                plots_dir:str,
                                val_accs:list,
                                train_accs:list,
                                lambdas:np.ndarray,
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

    PATH = os.path.join(plots_dir, 'grid_search')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, 'model_performances_over_time.png'))
    plt.close()

def plot_grid_search_results(
                            results:dict,
                            plot_dir:str,
                            rnd_seed:int,
                            modality:str,
                            version:str,
                            subfolder:str,
                            vision_model=None,
                            layer=None,
                            show_plot:bool=False,
) -> None:
    fig = plt.figure(figsize=(16, 8), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    lambdas = list(map(lambda l: round(float(l), 4), results.keys()))
    train_accs, val_accs = zip(*[(val['train_acc'], val['val_acc']) for lam, val in results.items()])

    ax.plot(train_accs, alpha=.8, label='Train')
    ax.plot(val_accs, alpha=.8, label='Val')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(lambdas)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel(r'$\lambda$')
    ax.legend(fancybox=True, shadow=True, loc='upper right')
    plt.tight_layout()

    if modality == 'visual':
        assert isinstance(vision_model, str) and isinstance(layer, str), 'name of vision model and corresponding layer are required'
        PATH = os.path.join(plot_dir, f'seed{rnd_seed}', modality, vision_model, layer, version, subfolder)
    else:
        PATH = os.path.join(plot_dir, f'seed{rnd_seed}', modality, version, subfolder)
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, 'lambda_search_results.png'))
    if show_plot:
        plt.show()
        plt.clf()
    plt.close()

def plot_aggregated_klds(
                        klds:np.ndarray,
                        plot_dir:str,
                        rnd_seed:int,
                        modality:str,
                        version:str,
                        dim:int,
                        lmbda:float,
                        reduction:str,
                        show_plot:bool=False,
                        ) -> None:
    """elbow plot of KL divergences aggregated over n_items"""
    fig = plt.figure(figsize=(16, 8), dpi=200)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.plot(klds)
    ax.set_xticks(np.arange(0, len(klds)+1, 10))
    ax.set_xlabel('Dimension', fontsize=10)
    ax.set_ylabel('KLD', fontsize=10)

    PATH = os.path.join(plot_dir, modality, version, f'{dim}d', f'{lmbda}', f'seed{rnd_seed}')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, f'kld_elbowplot_{reduction}.png'))

    if show_plot:
        plt.show()
        plt.clf()

    plt.close()

def plot_kld_violins(
                    klds:np.ndarray,
                    plot_dir:str,
                    rnd_seed:int,
                    modality:str,
                    version:str,
                    dim:int,
                    lmbda:float,
                    reduction:str,
                    show_plot:bool=False,
                    ) -> None:
    """violinplot of KL divergences across all items and latent dimensions"""
    fig = plt.figure(figsize=(16, 8), dpi=200)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.violinplot(klds, widths=0.8)
    ax.set_xticks(np.arange(0, klds.shape[1]+1, 10))
    ax.set_xlabel('Dimension', fontsize=10)
    ax.set_ylabel('KLD', fontsize=10)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)

    PATH = os.path.join(plot_dir, modality, version, f'{dim}d', f'{lmbda}',  f'seed{rnd_seed}')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, f'kld_violinplot_{reduction}.png'))

    if show_plot:
        plt.show()
        plt.clf()

    plt.close()

def plot_pruning_results(
                         results:list,
                         plot_dir:str,
                         rnd_seed:int,
                         modality:str,
                         version:str,
                         dim:int,
                         lmbda:float,
                         reduction:str,
                         ) -> None:
    """plot validation accuracy as a function of pruned weights percentage"""
    fig = plt.figure(figsize=(16, 8), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    pruning_fracs, val_accs = zip(*results)
    ax.bar(pruning_fracs, val_accs, alpha=.5, width=4.0)
    ax.set_xticks(pruning_fracs)
    ax.set_xticklabels(pruning_fracs)
    ax.set_ylim([np.floor(np.min(val_accs)), np.ceil(np.max(val_accs))])
    ax.set_ylabel('Val acc (%)')
    ax.set_xlabel(r'% of weights pruned')
    ax.set_title(f'$\lambda$ = {lmbda}')

    PATH = os.path.join(plot_dir, modality, version, f'{dim}d', f'{lmbda}', f'seed{rnd_seed}')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, f'val_acc_against_pruned_weights_{reduction}.png'))
    plt.close()
