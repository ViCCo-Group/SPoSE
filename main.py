#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import IPython; IPython.embed()
import argparse
import json
import logging
import os
import random
import re
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from scipy.stats import linregress
from torch.optim import Adam, AdamW
from tqdm import trange

from plotting import *
from utils import *
from models.model import *

os.environ['PYTHONIOENCODING']='UTF-8'
os.environ['CUDA_LAUNCH_BLOCKING']=str(1)

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--version', type=str, default='deterministic',
        choices=['deterministic', 'variational'],
        help='whether to apply a deterministic or variational version of SPoSE')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--folder', type=str, default='behavioral/',
        choices=['behavioral/', 'text/', 'visual/', 'neural/'],
        help='define for which modality task should be performed')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/)')
    aa('--tripletize', type=str, default=None,
        choices=[None, 'deterministic', 'probabilistic'],
        help='whether to deterministically (argmax) or probabilistically (conditioned on PMF) sample odd-one-out choices')
    aa('--beta', type=float, default=None,
        help='determines softmax temperature in probabilistic tripletizing approach')
    aa('--learning_rate', type=float, default=0.001,
        help='learning rate to be used in optimizer')
    aa('--lmbda', type=float,
        help='lambda value determines l1-norm fraction to regularize loss')
    aa('--embed_dim', metavar='D', type=int, default=90,
        help='dimensionality of the embedding matrix')
    aa('--batch_size', metavar='B', type=int, default=100,
        choices=[16, 25, 32, 50, 64, 100, 128, 150, 200, 256],
        help='number of triplets in each mini-batch')
    aa('--epochs', metavar='T', type=int, default=500,
        help='maximum number of epochs to optimize SPoSE model for')
    aa('--window_size', type=int, default=20,
        help='window size to be used for checking convergence criterion with linear regression')
    aa('--sampling_method', type=str, default='normal',
        choices=['normal', 'soft'],
        help='whether random sampling of the entire training set or soft sampling of some fraction of the training set will be performed during each epoch')
    aa('--p', type=float, default=None,
        choices=[None, 0.5, 0.6, 0.7, 0.8, 0.9],
        help='this argument is only necessary for soft sampling. specifies the fraction of *train* to be sampled during an epoch')
    aa('--plot_dims', action='store_true',
        help='whether or not to plot the number of non-negative dimensions as a function of time after convergence')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def setup_logging(file:str, dir:str='./log_files/'):
    #check whether directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)
    #create logger at root level (no need to provide specific name, as our logger won't have children)
    logger = logging.getLogger()
    logging.basicConfig(filename=dir + file, filemode='w', level=logging.DEBUG)
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
        version:str,
        task:str,
        seed:int,
        folder:str,
        device:torch.device,
        batch_size:int,
        embed_dim:int,
        epochs:int,
        window_size:int,
        sampling_method:str,
        lmbda:float,
        lr:float,
        p=None,
        embed_path=None,
        tripletize=None,
        beta=None,
        plot_dims:bool=True,
        show_progress:bool=True,
):
    #initialise logger and start logging events
    logger = setup_logging(file='spose_model_optimization.log')
    logger.setLevel(logging.INFO)

    #load train and test datasets
    if isinstance(tripletize, str):
        assert isinstance(embed_path, str), 'PATH from where to load neural activations or word embeddings must be defined'
        logger.info(f'Started tripletizing data with {tripletize} sampling of odd-one-out choices')
        if re.search(r'visual', folder):
            n_samples = 1.0e+7
            sampling_constant = 1.0e+6
        elif re.search(r'text', folder):
            n_samples = 1.5e+6
            sampling_constant = 1.5e+5
        train_triplets, test_triplets = tripletize_data(
                                                        PATH=embed_path,
                                                        method=tripletize,
                                                        n_samples=n_samples,
                                                        sampling_constant=sampling_constant,
                                                        beta=beta,
                                                        folder=folder,
                                                        device=device,
                                                        )
        logger.info('Finished tripletizing data')
    else:
        train_triplets, test_triplets = load_data(
                                                  device=device,
                                                  folder=folder,
                                                  )
    #number of unique items in the data matrix
    n_items = torch.max(train_triplets).item() + 1
    #initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    #create train and validation mini-batches
    train_batches = BatchGenerator(
                                    I=I,
                                    dataset=train_triplets,
                                    batch_size=batch_size,
                                    sampling_method=sampling_method,
                                    p=p,
                                    )
    val_batches = BatchGenerator(
                                 I=I,
                                 dataset=test_triplets,
                                 batch_size=batch_size,
                                 sampling_method=None,
                                 p=None,
                                 )

    ###############################
    ########## settings ###########
    ###############################

    #cutoff for significance (checking if slope is significantly decreasing)
    pval_thres = .1

    #l1-norm fraction to regularize loss
    lmbda /= n_items

    #initialize model and optimizer
    if version == 'variational':
        #variational version of SPoSE
        model = VSPoSE(in_size=n_items, out_size=embed_dim)
        k = 3 if task == 'odd_one_out' else 2
        mu = torch.zeros(batch_size * k, embed_dim).to(device)
        l = torch.ones(batch_size * k, embed_dim).mul(lmbda).to(device)
        optim = AdamW(model.parameters(), lr=lr)
        n_batches = len(train_batches) #for each mini-batch kld must be scaled by 1/B, where B = n_batches
    else:
        #deterministic version of SPoSE
        model = SPoSE(in_size=n_items, out_size=embed_dim, init_weights=True)
        optim = Adam(model.parameters(), lr=lr)

    #move model to current device
    model.to(device)

    model_path = os.path.join(results_dir, folder, version, str(lmbda),  'model')
    if os.path.exists(model_path):
        models = [m for m in os.listdir(model_path)]
        if len(models) > 0:
            try:
                checkpoints = list(map(lambda m: get_digits(m), models))
                last_checkpoint = np.argmax(checkpoints)
                PATH = os.path.join(model_path, models[last_checkpoint])
                checkpoint = torch.load(PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                optim.load_state_dict(checkpoint['optim_state_dict'])
                start = checkpoint['epoch'] #np.max(checkpoints)
                loss = checkpoint['loss']
                train_accs = checkpoint['train_accs']
                val_accs = checkpoint['val_accs']
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                nneg_d_over_time = checkpoint['nneg_d_over_time']
                print(f'...Loaded model and optimizer state dicts from previous run. Starting at epoch {start}.')
                print()
            except RuntimeError:
                print(f'...Loading model and optimizer state dicts failed. Check whether you are currently using a different set of model parameters.')
                print()
                start = 0
                train_accs, val_accs = [], []
                train_losses, val_losses = [], []
                nneg_d_over_time = []
        else:
            start = 0
            train_accs, val_accs = [], []
            train_losses, val_losses = [], []
            nneg_d_over_time = []
    else:
        start = 0
        train_accs, val_accs = [], []
        train_losses, val_losses = [], []
        nneg_d_over_time = []

    ################################################
    ################## training ####################
    ################################################

    iter = 0
    results = defaultdict(dict)
    logger.info(f'Optimization started for lambda: {lmbda}')

    for epoch in range(start, epochs):
        model.train()
        batch_losses_train = torch.zeros(len(train_batches))
        batch_accs_train = torch.zeros(len(train_batches))
        for i, batch in enumerate(train_batches):
            optim.zero_grad() #zero out gradients
            batch = batch.to(device)

            if version == 'variational':
                logits, z, mu_hat, l_hat = model(batch, device)
            else:
                logits = model(batch)

            anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, embed_dim)), dim=1)
            #TODO: figure out why the line below is necessary if we don't use the variable probs anywhere else in the script
            #probs = trinomial_probs(anchor, positive, negative, task)
            c_entropy = trinomial_loss(anchor, positive, negative, task)

            if version == 'variational':
                loss = c_entropy + (1/n_batches) * kl_div(mu, l, mu_hat, l_hat)
            else:
                l1_pen = l1_regularization(model).to(device) #L1-norm to enforce sparsity (many 0s)
                pos_pen = torch.sum(F.relu(-model.fc.weight)) #positivity constraint to enforce non-negative values in our embedding matrix
                loss = c_entropy + 0.01 * pos_pen + lmbda * l1_pen

            loss.backward() #backpropagate loss into the network
            optim.step() #compute gradients and update weights accordingly
            batch_losses_train[i] += loss.item()
            batch_accs_train[i] += choice_accuracy(anchor, positive, negative, task)
            iter += 1

        avg_train_loss = torch.mean(batch_losses_train)
        avg_train_acc = torch.mean(batch_accs_train)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        ################################################
        ################ validation ####################
        ################################################

        model.eval()
        with torch.no_grad():
            batch_losses_val = torch.zeros(len(val_batches))
            batch_accs_val = torch.zeros(len(val_batches))
            for j, batch in enumerate(val_batches):
                batch = batch.to(device)

                if version == 'variational':
                    logits, _, _, _ = model(batch, device)
                else:
                    logits = model(batch)

                anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, embed_dim)), dim=1)
                val_loss = trinomial_loss(anchor, positive, negative, task)
                batch_losses_val[j] += val_loss.item()
                batch_accs_val[j] += choice_accuracy(anchor, positive, negative, task)

        avg_val_loss = torch.mean(batch_losses_val)
        avg_val_acc = torch.mean(batch_accs_val)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        logger.info('Epoch: {0}/{1}'.format(epoch + 1, epochs))
        logger.info('Train acc: {:.3f}'.format(avg_train_acc))
        logger.info('Train loss: {:.3f}'.format(avg_train_loss))
        logger.info('Val acc: {:.3f}'.format(avg_val_acc))
        logger.info('Val loss: {:.3f}'.format(avg_val_loss))

        if show_progress:
            print("========================================================================================================")
            print('====== Epoch: {0}, Train acc: {1:.3f}, Train loss: {2:.3f}, Val acc: {3:.3f}, Val loss: {4:.3f} ======'.format(epoch + 1, avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss))
            print("========================================================================================================")

        if (epoch + 1) % 5 == 0:
            PATH = os.path.join(results_dir, folder, version, str(lmbda))
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            if version == 'deterministic':
                W = model.fc.weight
                np.savetxt(os.path.join(PATH, f'sparse_embed_epoch{epoch+1:04d}.txt'), W.detach().cpu().numpy())
                logger.info(f'Saving model weights at epoch {epoch+1}')

                if plot_dims:
                    current_d = get_nneg_dims(W)
                    nneg_d_over_time.append((epoch+1, current_d))

            PATH = os.path.join(PATH, 'model')
            if not os.path.exists(PATH):
                os.makedirs(PATH)

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
                        }, os.path.join(PATH, f'model_epoch{epoch+1:04d}.tar'))
            logger.info(f'Saving model parameters at epoch {epoch+1}')

            if (epoch + 1) > window_size:
                #check termination condition
                lmres = linregress(range(window_size), train_losses[(epoch + 1 - window_size):(epoch + 2)])
                if (lmres.slope > 0) or (lmres.pvalue > pval_thres):
                    break

    results[lmbda] = {
                      'epoch': int(np.argmax(val_accs)+1),
                      'train_acc': float(train_accs[np.argmax(val_accs)]),
                      'val_acc': float(np.max(val_accs)),
                      'val_loss': float(np.min(val_losses)),
                      }
    logger.info(f'Optimization finished after {epoch+1} epochs for lambda: {lmbda}')

    if plot_dims:
        if version == 'deterministic':
            logger.info(f'Plotting number of non-negative dimensions as a function of time for lambda: {lmbda}')
            plot_nneg_dims_over_time(nneg_d_over_time, lmbda, folder, version)

    logger.info('Plotting model performances over time across all lambda values')
    plot_single_performance(val_accs, train_accs, lmbda, folder, version)

    PATH = os.path.join(results_dir, folder, version, 'hyperparams')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    with open(os.path.join(PATH, 'lambda_search' + '_' + str(lmbda) + '.json'), 'w') as results_file:
        json.dump(results, results_file)

if __name__ == "__main__":
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

    results_dir = os.path.join(args.results_dir, f'seed{args.rnd_seed:02d}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if isinstance(args.tripletize, str):
        if re.search(r'text', args.folder):
            #NOTE: if you have an embedding matrix (i.e., an embedding per object), you can automatically tripletize the data (simply change .csv file)
            embed_path = os.path.join(args.folder, 'sensevec.csv')
        elif re.search(r'visual', args.folder):
            #NOTE: if you have a matrix of hidden unit activations per object, you can automatically tripletize the data (simply change .txt file)
            embed_path = os.path.join(args.folder, 'activations.txt')
    else:
        embed_path = None

    run(
        version=args.version,
        task=args.task,
        seed=args.rnd_seed,
        folder=args.folder,
        device=device,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        window_size=args.window_size,
        sampling_method=args.sampling_method,
        lmbda=args.lmbda,
        lr=args.learning_rate,
        p=args.p,
        embed_path=embed_path,
        tripletize=args.tripletize,
        beta=args.beta,
        plot_dims=args.plot_dims,
        )
