#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Max Planck Institute for Human Cognitive and Brain Sciences
# Author: Lukas Muttenthaler

import argparse
import json
import logging
import os
import random
import re
from turtle import distance
import torch
import warnings
import utils

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from collections import defaultdict
from scipy.stats import linregress
from torch.optim import Adam, AdamW

from plotting import *
from models.model import *

os.environ['PYTHONIOENCODING']='UTF-8'
os.environ['CUDA_LAUNCH_BLOCKING']=str(1)

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--modality', type=str, default='behavioral/',
        #choices=['behavioral/', 'text/', 'visual/', 'neural/'],
        help='define for which modality SPoSE should be perform specified task')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/modality/lambda/rnd_seed/)')
    aa('--plots_dir', type=str, default='./plots/',
        help='optional specification of directory for plots (if not provided will resort to ./plots/modality/lambda/rnd_seed/)')
    aa('--learning_rate', type=float, default=0.001,
        help='learning rate to be used in optimizer')
    aa('--lmbda', type=float,
        help='lambda value determines weight of l1-regularization')
    aa('--temperature', type=float, default=1.,
        help='softmax temperature (beta param) for choice randomness')
    aa('--embed_dim', metavar='D', type=int, default=90,
        help='dimensionality of the embedding matrix')
    aa('--batch_size', metavar='B', type=int, default=100,
        choices=[16, 25, 32, 50, 64, 100, 128, 150, 200, 256],
        help='number of triplets in each mini-batch')
    aa('--epochs', metavar='T', type=int, default=500,
        help='maximum number of epochs to optimize SPoSE model for')
    aa('--window_size', type=int, default=50,
        help='window size to be used for checking convergence criterion with linear regression')
    aa('--steps', type=int, default=10,
        help='save model parameters and create checkpoints every <steps> epochs')
    aa('--sampling_method', type=str, default='normal',
        choices=['normal', 'soft'],
        help='whether random sampling of the entire training set or soft sampling of some fraction of the training set will be performed during each epoch')
    aa('--p', type=float, default=None,
        choices=[None, 0.5, 0.6, 0.7, 0.8, 0.9],
        help='this argument is only necessary for soft sampling. specifies the fraction of *train* to be sampled during an epoch')
    aa('--resume', action='store_true',
        help='whether to resume training at last checkpoint; if not set training will restart')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    aa('--distance_metric', type=str, default='dot', choices=['dot', 'euclidean'], help='distance metric')
    args = parser.parse_args()
    return args

def setup_logging(file:str, dir:str='./log_files/'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    #create logger at root level (no need to provide specific name, as our logger won't have children)
    logger = logging.getLogger()
    logging.basicConfig(filename=os.path.join(dir, file), filemode='w', level=logging.DEBUG)
    #add console handler to logger
    if len(logger.handlers) < 1:
        #create console handler and set level to debug (lowest severity level)
        handler = logging.StreamHandler()
        #this specifies the lowest-severity log message the logger will handle
        handler.setLevel(logging.DEBUG)
        #create formatter to configure order, structure, and content of log messages
        formatter = logging.Formatter(fmt="%(asctime)s - [%(levelname)s] - %(message)s", datefmt='%d/%m/%Y %I:%M:%S %p')
        #add formatter to handler
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def run(
        task:str,
        rnd_seed:int,
        modality:str,
        results_dir:str,
        plots_dir:str,
        triplets_dir:str,
        device:torch.device,
        batch_size:int,
        embed_dim:int,
        epochs:int,
        window_size:int,
        sampling_method:str,
        lmbda:float,
        lr:float,
        steps:int,
        p:float=None,
        resume:bool=False,
        show_progress:bool=True,
        distance_metric:str='dot',
        temperature:float=1.
):
    #initialise logger and start logging events
    logger = setup_logging(file='spose_optimization.log', dir=f'./log_files/lmbda_{lmbda}/')
    logger.setLevel(logging.INFO)
    #load triplets into memory
    train_triplets, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir)
    n_items = utils.get_nitems(train_triplets)
    #load train and test mini-batches
    train_batches, val_batches = utils.load_batches(
                                                      train_triplets=train_triplets,
                                                      test_triplets=test_triplets,
                                                      n_items=n_items,
                                                      batch_size=batch_size,
                                                      sampling_method=sampling_method,
                                                      rnd_seed=rnd_seed,
                                                      p=p,
                                                      )
    print(f'\nNumber of train batches in current process: {len(train_batches)}\n')

    ###############################
    ########## settings ###########
    ###############################

    temperature = torch.tensor(temperature).to(device)
    model = SPoSE(in_size=n_items, out_size=embed_dim, init_weights=True)
    model.to(device)
    optim = Adam(model.parameters(), lr=lr)

    ################################################
    ############# Creating PATHs ###################
    ################################################

    print(f'...Creating PATHs')
    print()
    if results_dir == './results/':
        results_dir = os.path.join(results_dir, modality, f'{embed_dim}d', str(lmbda), f'seed{rnd_seed:02d}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if plots_dir == './plots/':
        plots_dir = os.path.join(plots_dir, modality, f'{embed_dim}d', str(lmbda), f'seed{rnd_seed}')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    model_dir = os.path.join(results_dir, 'model')

    #####################################################################
    ######### Load model from previous checkpoint, if available #########
    #####################################################################

    if resume:
        if os.path.exists(model_dir):
            models = sorted([m.name for m in os.scandir(model_dir) if m.name.endswith('.tar')])
            if len(models) > 0:
                try:
                    PATH = os.path.join(model_dir, models[-1])
                    map_location = device
                    checkpoint = torch.load(PATH, map_location=map_location)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optim.load_state_dict(checkpoint['optim_state_dict'])
                    start = checkpoint['epoch'] + 1
                    loss = checkpoint['loss']
                    train_accs = checkpoint['train_accs']
                    val_accs = checkpoint['val_accs']
                    train_losses = checkpoint['train_losses']
                    val_losses = checkpoint['val_losses']
                    nneg_d_over_time = checkpoint['nneg_d_over_time']
                    loglikelihoods = checkpoint['loglikelihoods']
                    complexity_losses = checkpoint['complexity_costs']
                    print(f'...Loaded model and optimizer state dicts from previous run. Starting at epoch {start}.\n')
                except RuntimeError:
                    print(f'...Loading model and optimizer state dicts failed. Check whether you are currently using a different set of model parameters.\n')
                    start = 0
                    train_accs, val_accs = [], []
                    train_losses, val_losses = [], []
                    loglikelihoods, complexity_losses = [], []
                    nneg_d_over_time = []
            else:
                raise Exception('No checkpoints found. Cannot resume training.')
        else:
            raise Exception('Model directory does not exist. Cannot resume training.')
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        start = 0
        train_accs, val_accs = [], []
        train_losses, val_losses = [], []
        loglikelihoods, complexity_losses = [], []
        nneg_d_over_time = []

    ################################################
    ################## Training ####################
    ################################################

    iter = 0
    results = {}
    logger.info(f'Optimization started for lambda: {lmbda}\n')

    print(f'Optimization started for lambda: {lmbda}\n')
    for epoch in range(start, epochs):
        model.train()
        batch_llikelihoods = torch.zeros(len(train_batches))
        batch_closses = torch.zeros(len(train_batches))
        batch_losses_train = torch.zeros(len(train_batches))
        batch_accs_train = torch.zeros(len(train_batches))
        for i, batch in enumerate(train_batches):
            optim.zero_grad() #zero out gradients
            batch = batch.to(device)
            logits = model(batch)
            anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, embed_dim)), dim=1)
            c_entropy = utils.trinomial_loss(anchor, positive, negative, task, temperature, distance_metric)
            l1_pen = l1_regularization(model).to(device) #L1-norm to enforce sparsity (many 0s)
            W = model.fc.weight
            pos_pen = torch.sum(F.relu(-W)) #positivity constraint to enforce non-negative values in embedding matrix
            complexity_loss = (lmbda/n_items) * l1_pen
            loss = c_entropy + 0.01 * pos_pen + complexity_loss
            loss.backward()
            optim.step()
            batch_losses_train[i] += loss.item()
            batch_llikelihoods[i] += c_entropy.item()
            batch_closses[i] += complexity_loss.item()
            batch_accs_train[i] += utils.choice_accuracy(anchor, positive, negative, task, distance_metric)
            iter += 1

        avg_llikelihood = torch.mean(batch_llikelihoods).item()
        avg_closs = torch.mean(batch_closses).item()
        avg_train_loss = torch.mean(batch_losses_train).item()
        avg_train_acc = torch.mean(batch_accs_train).item()

        loglikelihoods.append(avg_llikelihood)
        complexity_losses.append(avg_closs)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        ################################################
        ################ validation ####################
        ################################################

        avg_val_loss, avg_val_acc = utils.validation(model=model, val_batches=val_batches, task=task, device=device, distance_metric=distance_metric)

        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        logger.info(f'Epoch: {epoch+1}/{epochs}')
        logger.info(f'Train acc: {avg_train_acc:.5f}')
        logger.info(f'Train loss: {avg_train_loss:.5f}')
        logger.info(f'Val acc: {avg_val_acc:.5f}')
        logger.info(f'Val loss: {avg_val_loss:.5f}\n')

        if show_progress:
            print("\n========================================================================================================")
            print(f'====== Epoch: {epoch+1}, Train acc: {avg_train_acc:.5f}, Train loss: {avg_train_loss:.5f}, Val acc: {avg_val_acc:.5f}, Val loss: {avg_val_loss:.5f} ======')
            print("========================================================================================================\n")
            current_d = utils.get_nneg_dims(W)
            nneg_d_over_time.append((epoch+1, current_d))
            print("\n========================================================================================================")
            print(f"========================= Current number of non-negative dimensions: {current_d} =========================")
            print("========================================================================================================\n")

        if (epoch + 1) % steps == 0:
            W = model.fc.weight
            np.savetxt(os.path.join(results_dir, f'sparse_embed_epoch{epoch+1:04d}.txt'), W.detach().cpu().numpy())
            logger.info(f'Saving model weights at epoch {epoch+1}')

            #save model and optim parameters for inference or to resume training
            #PyTorch convention is to save checkpoints as .tar files
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optim.state_dict(),
                        'loss': loss,
                        'train_losses': train_losses,
                        'train_accs': train_accs,
                        'val_losses': val_losses,
                        'val_accs': val_accs,
                        'nneg_d_over_time': nneg_d_over_time,
                        'loglikelihoods': loglikelihoods,
                        'complexity_costs': complexity_losses,
                        }, os.path.join(model_dir, f'model_epoch{epoch+1:04d}.tar'))

            logger.info(f'Saving model parameters at epoch {epoch+1}\n')

        if (epoch + 1) > window_size:
            #check termination condition (we want to train until convergence)
            lmres = linregress(range(window_size), train_losses[(epoch + 1 - window_size):(epoch + 2)])
            if (lmres.slope > 0) or (lmres.pvalue > .1):
                break

    #save final model weights
    utils.save_weights_(results_dir, model.fc.weight)
    results = {'epoch': len(train_accs), 'train_acc': train_accs[-1], 'val_acc': val_accs[-1], 'val_loss': val_losses[-1]}
    logger.info(f'\nOptimization finished after {epoch+1} epochs for lambda: {lmbda}\n')

    logger.info(f'\nPlotting number of non-negative dimensions as a function of time for lambda: {lmbda}\n')
    plot_nneg_dims_over_time(plots_dir=plots_dir, nneg_d_over_time=nneg_d_over_time)

    logger.info(f'\nPlotting model performances over time for lambda: {lmbda}')
    #plot train and validation performance alongside each other to examine a potential overfit to the training data
    plot_single_performance(plots_dir=plots_dir, val_accs=val_accs, train_accs=train_accs)
    logger.info(f'\nPlotting losses over time for lambda: {lmbda}')
    #plot both log-likelihood of the data (i.e., cross-entropy loss) and complexity loss (i.e., l1-norm in DSPoSE and KLD in VSPoSE)
    plot_complexities_and_loglikelihoods(plots_dir=plots_dir, loglikelihoods=loglikelihoods, complexity_losses=complexity_losses)

    PATH = os.path.join(results_dir, 'results.json')
    with open(PATH, 'w') as results_file:
        json.dump(results, results_file)

if __name__ == "__main__":
    #parse all arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    if re.search(r'^cuda', args.device):
        device = torch.device(args.device)
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        try:
            torch.cuda.set_device(int(args.device[-1]))
        except:
            torch.cuda.set_device(1)
        print(f'\nPyTorch CUDA version: {torch.version.cuda}\n')
    else:
        device = torch.device(args.device)

    run(
        task=args.task,
        rnd_seed=args.rnd_seed,
        modality=args.modality,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        triplets_dir=args.triplets_dir,
        device=device,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        window_size=args.window_size,
        sampling_method=args.sampling_method,
        lmbda=args.lmbda,
        lr=args.learning_rate,
        steps=args.steps,
        resume=args.resume,
        p=args.p,
        distance_metric=args.distance_metric,
        temperature=args.temperature
        )
