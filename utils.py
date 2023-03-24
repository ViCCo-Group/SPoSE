#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'BatchGenerator',
            'TripletDataset',
            'choice_accuracy',
            'cross_entropy_loss',
            'compute_kld',
            'compare_modalities',
            'corr_mat',
            'compute_trils',
            'cos_mat',
            'cross_correlate_latent_dims',
            'encode_as_onehot',
            'get_cut_off',
            'get_digits',
            'get_nneg_dims',
            'get_ref_indices',
            'get_results_files',
            'get_nitems',
            'kld_online',
            'kld_offline',
            'load_batches',
            'load_concepts',
            'load_data',
            'load_inds_and_item_names',
            'load_model',
            'load_sparse_codes',
            'load_ref_images',
            'load_targets',
            'load_weights',
            'l2_reg_',
            'matmul',
            'merge_dicts',
            'pickle_file',
            'unpickle_file',
            'pearsonr',
            'prune_weights',
            'rsm',
            'rsm_pred',
            'save_weights_',
            'sparsity',
            'spose2rsm_odd_one_out',
            'avg_sparsity',
            'softmax',
            'sort_weights',
            'trinomial_loss',
            'trinomial_probs',
            'validation',
        ]

import json
import logging
import math
import os
import pickle
import re
import torch
import warnings

import numpy as np
import pandas as pd
import skimage.io as io
import torch.nn.functional as F

from collections import defaultdict, Counter
from itertools import combinations, permutations
from numba import njit, jit, prange
from os.path import join as pjoin
from skimage.transform import resize
from torch.optim import Adam, AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import Tuple, Iterator, List, Dict

class TripletDataset(Dataset):

    def __init__(self, I:torch.tensor, dataset:torch.Tensor):
        self.I = I
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx:int) -> torch.Tensor:
        sample = encode_as_onehot(self.I, self.dataset[idx])
        return sample

class BatchGenerator(object):

    def __init__(
                self,
                I:torch.tensor,
                dataset:torch.Tensor,
                batch_size:int,
                sampling_method:str='normal',
                p=None,
):
        self.I = I
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampling_method = sampling_method
        self.p = p

        if sampling_method == 'soft':
            assert isinstance(self.p, float)
            self.n_batches = int(len(self.dataset) * self.p) // self.batch_size
        else:
            self.n_batches = len(self.dataset) // self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_batches(self.I, self.dataset)

    def sampling(self, triplets:torch.Tensor) -> torch.Tensor:
        """randomly sample training data during each epoch"""
        rnd_perm = torch.randperm(len(triplets))
        if self.sampling_method == 'soft':
            rnd_perm = rnd_perm[:int(len(rnd_perm) * self.p)]
        return triplets[rnd_perm]

    def get_batches(self, I:torch.Tensor, triplets:torch.Tensor) -> Iterator[torch.Tensor]:
        if not isinstance(self.sampling_method, type(None)):
            triplets = self.sampling(triplets)
        for i in range(self.n_batches):
            batch = encode_as_onehot(I, triplets[i*self.batch_size: (i+1)*self.batch_size])
            yield batch

def pickle_file(file:dict, out_path:str, file_name:str) -> None:
    with open(os.path.join(out_path, ''.join((file_name, '.txt'))), 'wb') as f:
        f.write(pickle.dumps(file))

def unpickle_file(in_path:str, file_name:str) -> dict:
    return pickle.loads(open(os.path.join(in_path, ''.join((file_name, '.txt'))), 'rb').read())

def assert_nneg(X:np.ndarray, thresh:float=1e-5) -> np.ndarray:
    """if data matrix X contains negative real numbers, transform matrix into R+ (i.e., positive real number(s) space)"""
    if np.any(X < 0):
        X -= np.amin(X, axis=0)
        return X + thresh
    return X

def load_inds_and_item_names(folder:str='./data') -> Tuple[np.ndarray]:
    item_names = pd.read_csv(pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values
    sortindex = pd.read_table(pjoin(folder, 'sortindex'), header=None)[0].values
    return item_names, sortindex

def load_ref_images(img_folder:str, item_names:np.ndarray) -> np.ndarray:
    ref_images = np.array([resize(io.imread(pjoin('./reference_images', name + '.jpg')), (400, 400), anti_aliasing=True) for name in item_names])
    return ref_images

def load_concepts(folder:str='./data') -> pd.DataFrame:
    concepts = pd.read_csv(pjoin(folder, 'category_mat_manual.tsv'), encoding='utf-8', sep='\t')
    return concepts

def load_data(device:torch.device, triplets_dir:str, inference:bool=False) -> Tuple[torch.Tensor]:
    """load train and test triplet datasets into memory"""
    if inference:
        with open(pjoin(triplets_dir, 'test_triplets.npy'), 'rb') as test_triplets:
            test_triplets = torch.from_numpy(np.load(test_triplets)).to(device).type(torch.LongTensor)
            return test_triplets
    try:
        with open(pjoin(triplets_dir, 'train_90.npy'), 'rb') as train_file:
            train_triplets = torch.from_numpy(np.load(train_file)).to(device).type(torch.LongTensor)

        with open(pjoin(triplets_dir, 'test_10.npy'), 'rb') as test_file:
            test_triplets = torch.from_numpy(np.load(test_file)).to(device).type(torch.LongTensor)
    except FileNotFoundError:
        print('\n...Could not find any .npy files for current modality.')
        print('...Now searching for .txt files.\n')
        train_triplets = torch.from_numpy(np.loadtxt(pjoin(triplets_dir, 'train_90.txt'))).to(device).type(torch.LongTensor)
        test_triplets = torch.from_numpy(np.loadtxt(pjoin(triplets_dir, 'test_10.txt'))).to(device).type(torch.LongTensor)
    return train_triplets, test_triplets

def get_nitems(train_triplets:torch.Tensor) -> int:
    #number of unique items in the data matrix
    n_items = torch.max(train_triplets).item()
    if torch.min(train_triplets).item() == 0:
        n_items += 1
    return n_items

def load_batches(
                 train_triplets:torch.Tensor,
                 test_triplets:torch.Tensor,
                 n_items:int,
                 batch_size:int,
                 inference:bool=False,
                 sampling_method:str=None,
                 rnd_seed:int=None,
                 multi_proc:bool=False,
                 n_gpus:int=None,
                 p=None,
                 ):
    #initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    if inference:
        assert train_triplets is None
        test_batches = BatchGenerator(I=I, dataset=test_triplets, batch_size=batch_size, sampling_method=None, p=None)
        return test_batches
    if (multi_proc and n_gpus > 1):
        if sampling_method == 'soft':
            warnings.warn(f'...Soft sampling cannot be used in a multi-process distributed training setting.', RuntimeWarning)
            warnings.warn(f'...Processes will equally distribute the entire training dataset amongst each other.', RuntimeWarning)
            warnings.warn(f'...If you want to use soft sampling, you must switch to single GPU or CPU training.', UserWarning)
        train_set = TripletDataset(I=I, dataset=train_triplets)
        val_set = TripletDataset(I=I, dataset=test_triplets)
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True, seed=rnd_seed)
        train_batches = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=n_gpus)
        val_batches = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=n_gpus)
    else:
        #create two iterators of train and validation mini-batches respectively
        train_batches = BatchGenerator(I=I, dataset=train_triplets, batch_size=batch_size, sampling_method=sampling_method, p=p)
        val_batches = BatchGenerator(I=I, dataset=test_triplets, batch_size=batch_size, sampling_method=None, p=None)
    return train_batches, val_batches

def l2_reg_(model, weight_decay:float=1e-5) -> torch.Tensor:
    loc_norms_squared = .5 * (model.encoder_mu[0].weight.pow(2).sum() + model.encoder_mu[0].bias.pow(2).sum())
    scale_norms_squared = (model.encoder_b[0].weight.pow(2).sum() +  model.encoder_mu[0].bias.pow(2).sum())
    l2_reg = weight_decay * (loc_norms_squared + scale_norms_squared)
    return l2_reg

def encode_as_onehot(I:torch.Tensor, triplets:torch.Tensor) -> torch.Tensor:
    """encode item triplets as one-hot-vectors"""
    return I[triplets.flatten(), :]

def softmax(sims:tuple, t:torch.Tensor) -> torch.Tensor:
    return torch.exp(sims[0] / t) / torch.sum(torch.stack([torch.exp(sim / t) for sim in sims]), dim=0)

def cross_entropy_loss(sims:tuple, t:torch.Tensor) -> torch.Tensor:
    return torch.mean(-torch.log(softmax(sims, t)))

def compute_similarities(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str, distance_metric:str = 'dot') -> Tuple:
    if distance_metric == 'dot':
        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)
        if method == 'odd_one_out':
            neg_sim_2 = torch.sum(positive * negative, dim=1)
            return pos_sim, neg_sim, neg_sim_2
        else:
            return pos_sim, neg_sim
    elif distance_metric == 'euclidean':
        pos_sim = -1*torch.sqrt(torch.sum(torch.square(torch.sub(anchor,positive)), dim=1))
        neg_sim = -1*torch.sqrt(torch.sum(torch.square(torch.sub(anchor,negative)), dim=1))
        
        if method == 'odd_one_out':
            neg_sim_2 = -1*torch.sqrt(torch.sum(torch.square(torch.sub(positive,negative)), dim=1))
            return pos_sim, neg_sim, neg_sim_2
        else:
            return pos_sim, neg_sim

def accuracy_(probas:torch.Tensor) -> float:
    choices = np.where(probas.mean(axis=1) == probas.max(axis=1), -1, np.argmax(probas, axis=1))
    acc = np.where(choices == 0, 1, 0).mean()
    return acc

def choice_accuracy(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str, distance_metric: str = 'dot') -> float:
    similarities  = compute_similarities(anchor, positive, negative, method, distance_metric)
    probas = F.softmax(torch.stack(similarities, dim=-1), dim=1).detach().cpu().numpy()
    return accuracy_(probas)

def trinomial_probs(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str, t:torch.Tensor, distance_metric: str = 'dot') -> torch.Tensor:
    sims = compute_similarities(anchor, positive, negative, method, distance_metric)
    return softmax(sims, t)

def trinomial_loss(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str, t:torch.Tensor, distance_metric: str = 'dot') -> torch.Tensor:
    sims = compute_similarities(anchor, positive, negative, method, distance_metric)
    return cross_entropy_loss(sims, t)

def kld_online(mu_1:torch.Tensor, l_1:torch.Tensor, mu_2:torch.Tensor, l_2:torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(l_1/l_2) + (l_2/l_1) * torch.exp(-l_1 * torch.abs(mu_1-mu_2)) + l_2*torch.abs(mu_1-mu_2) - 1)

def kld_offline(mu_1:torch.Tensor, b_1:torch.Tensor, mu_2:torch.Tensor, b_2:torch.Tensor) -> torch.Tensor:
    return torch.log(b_2/b_1) + (b_1/b_2) * torch.exp(-torch.abs(mu_1-mu_2)/b_1) + torch.abs(mu_1-mu_2)/b_2 - 1

def get_nneg_dims(W:torch.Tensor, eps:float=0.1) -> int:
    w_max = W.max(dim=1)[0]
    nneg_d = len(w_max[w_max > eps])
    return nneg_d

def remove_zeros(W:np.ndarray, eps:float=.1) -> np.ndarray:
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W

################################################
######### helper functions for evaluation ######
################################################

def get_seeds(PATH:str) -> List[str]:
    return [dir.name for dir in os.scandir(PATH) if dir.is_dir() and dir.name.startswith('seed')]

def seed_sampling(PATH:str) -> np.ndarray:
    seed = np.random.choice(get_seeds(PATH))
    with open(os.path.join(PATH, seed, 'test_probas.npy'), 'rb') as f:
        probas = np.load(f)
    return probas

def instance_sampling(probas:np.ndarray) -> np.ndarray:
    rnd_sample = np.random.choice(np.arange(len(probas)), size=len(probas), replace=True)
    probas_draw = probas[rnd_sample]
    return probas_draw

def get_global_averages(avg_probas:dict) -> np.ndarray:
    sorted_bins = dict(sorted(avg_probas.items()))
    return np.array([np.mean(p) for p in sorted_bins.values()])

def compute_pm(probas:np.ndarray) -> Tuple[np.ndarray, dict]:
    """compute probability mass for every choice"""
    avg_probas = defaultdict(list)
    count_vector = np.zeros((2, 11))
    for pmf in probas:
        indices = np.round(pmf*10).astype(int)
        count_vector[0, indices[0]] += 1
        count_vector[1, indices] += 1
        for k, p in enumerate(pmf):
            avg_probas[int(indices[k])].append(p)
    model_confidences = count_vector[0]/count_vector[1]
    avg_probas = get_global_averages(avg_probas)
    return model_confidences, avg_probas

def mse(avg_p:np.ndarray, confidences:np.ndarray) -> float:
    return np.mean((avg_p - confidences)**2)

def bootstrap_calibrations(PATH:str, alpha:float, n_bootstraps:int=1000) -> np.ndarray:
    mses = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        probas = seed_sampling(PATH)
        probas_draw = instance_sampling(probas)
        confidences, avg_p = compute_pm(probas_draw, alpha)
        mses[i] += mse(avg_p, confidences)
    return mses

def get_model_confidence_(PATH:str) -> Tuple[np.ndarray, np.ndarray]:
    seeds = get_seeds(PATH)
    confidence_scores = np.zeros((len(seeds), 11))
    avg_probas = np.zeros((len(seeds), 11))
    for i, seed in enumerate(seeds):
        with open(os.path.join(PATH, seed, 'test_probas.npy'), 'rb') as f:
            confidence, avg_p = compute_pm(np.load(f))
            confidence_scores[i] += confidence
            avg_probas[i] += avg_p
    return confidence_scores, avg_probas

def smoothing_(p:np.ndarray, alpha:float=.1) -> np.ndarray:
    return (p + alpha) / np.sum(p + alpha)

def entropy_(p:np.ndarray) -> np.ndarray:
    return np.sum(np.where(p == 0, 0, p*np.log(p)))

def cross_entropy_(p:np.ndarray, q:np.ndarray, alpha:float) -> float:
    return -np.sum(p*np.log(smoothing_(q, alpha)))

def kld_(p:np.ndarray, q:np.ndarray, alpha:float) -> float:
    return entropy_(p) + cross_entropy_(p, q, alpha)

def compute_divergences(human_pmfs:dict, model_pmfs:dict, metric:str='kld') -> np.ndarray:
    assert len(human_pmfs) == len(model_pmfs), '\nNumber of triplets in human and model distributions must correspond.\n'
    divergences = np.zeros(len(human_pmfs))
    for i, (triplet, p) in enumerate(human_pmfs.items()):
        q = model_pmfs[triplet]
        div = kld_(p, q) if metric  == 'kld' else cross_entropy_(p, q)
        divergences[i] += div
    return divergences

def mat2py(triplet:tuple) -> tuple:
    return tuple(np.asarray(triplet)-1)

def pmf(hist:dict) -> np.ndarray:
    values = np.array(list(hist.values()))
    return values/np.sum(values)

def histogram(choices:list, behavior:bool=False) -> dict:
    hist = {i+1 if behavior else i: 0 for i in range(3)}
    for choice in choices:
        hist[choice if behavior else choice.item()] += 1
    return hist

def compute_pmfs(choices:dict, behavior:bool) -> dict:
    pmfs = {mat2py(t) if behavior else t: pmf(histogram(c, behavior)) for t, c in choices.items()}
    return pmfs

def get_choice_distributions(test_set:pd.DataFrame) -> dict:
    """function to compute human choice distributions and corresponding pmfs"""
    triplets = test_set[['trip.1', 'trip.2', 'trip.3']]
    test_set['triplets'] = list(map(tuple, triplets.to_numpy()))
    unique_triplets = test_set.triplets.unique()
    choice_distribution = defaultdict(list)
    for triplet in unique_triplets:
        choices = list(test_set[test_set['triplets']==triplet].choice.values)
        sorted_choices = [np.where(np.argsort(triplet)+1==c)[0][0]+1 for c in choices]
        sorted_triplet = tuple(sorted(triplet))
        choice_distribution[sorted_triplet].extend(sorted_choices)
    choice_pmfs = compute_pmfs(choice_distribution, behavior=True)
    return choice_pmfs

def collect_choices(probas:np.ndarray, human_choices:np.ndarray, model_choices:dict) -> dict:
    """collect model choices at inference time"""
    probas = probas.flip(dims=[1])
    for pmf, choices in zip(probas, human_choices):
        sorted_choices = tuple(np.sort(choices))
        model_choices[sorted_choices].append(np.argmax(pmf[np.argsort(choices)]))
    return model_choices

def logsumexp_(logits:torch.Tensor) -> torch.Tensor:
    return torch.exp(logits - torch.logsumexp(logits, dim=1)[..., None])

def test(
        model,
        test_batches,
        version:str,
        task:str,
        device:torch.device,
        batch_size=None,
        n_samples=None,
        distance_metric: str = 'dot',
        temperature:float=1.
) -> Tuple:
    probas = torch.zeros(int(len(test_batches) * batch_size), 3)
    temperature = torch.tensor(temperature).to(device)
    model_choices = defaultdict(list)
    model.eval()
    with torch.no_grad():
        batch_accs = torch.zeros(len(test_batches))
        for j, batch in enumerate(test_batches):
            batch = batch.to(device)
            if version == 'variational':
                assert isinstance(n_samples, int), '\nOutput logits of variational neural networks have to be averaged over different samples through mc sampling.\n'
                test_acc, _, batch_probas = mc_sampling(model=model, batch=batch, temperature=temperature, task=task, n_samples=n_samples, device=device)
            else:
                logits = model(batch)
                anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)
                similarities = compute_similarities(anchor, positive, negative, task, distance_metric)
                #stacked_sims = torch.stack(similarities, dim=-1)
                #batch_probas = F.softmax(logsumexp_(stacked_sims), dim=1)
                batch_probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
                test_acc = choice_accuracy(anchor, positive, negative, task)

            probas[j*batch_size:(j+1)*batch_size] += batch_probas
            batch_accs[j] += test_acc
            human_choices = batch.nonzero(as_tuple=True)[-1].view(batch_size, -1).numpy()
            model_choices = collect_choices(batch_probas, human_choices, model_choices)

    probas = probas.cpu().numpy()
    probas = probas[np.where(probas.sum(axis=1) != 0.)]
    model_pmfs = compute_pmfs(model_choices, behavior=False)
    test_acc = batch_accs.mean().item()
    return test_acc, probas, model_pmfs

def validation(
                model,
                val_batches,
                task:str,
                device:torch.device,
                sampling:bool=False,
                batch_size=None,
                distance_metric: str = 'dot'
                ):
    if sampling:
        assert isinstance(batch_size, int), 'batch size must be defined'
        sampled_choices = np.zeros((int(len(val_batches) * batch_size), 3), dtype=int)

    temperature = torch.tensor(1.).to(device)
    model.eval()
    with torch.no_grad():
        batch_losses_val = torch.zeros(len(val_batches))
        batch_accs_val = torch.zeros(len(val_batches))
        for j, batch in enumerate(val_batches):
            batch = batch.to(device)
            logits = model(batch)
            anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)

            if sampling:
                similarities = compute_similarities(anchor, positive, negative, task, distance_metric)
                probas = F.softmax(torch.stack(similarities, dim=-1), dim=1).numpy()
                probas = probas[:, ::-1]
                human_choices = batch.nonzero(as_tuple=True)[-1].view(batch_size, -1).numpy()
                model_choices = np.array([np.random.choice(h_choice, size=len(p), replace=False, p=p)[::-1] for h_choice, p in zip(human_choices, probas)])
                sampled_choices[j*batch_size:(j+1)*batch_size] += model_choices
            else:
                val_loss = trinomial_loss(anchor, positive, negative, task, temperature)
                val_acc = choice_accuracy(anchor, positive, negative, task)

            batch_losses_val[j] += val_loss.item()
            batch_accs_val[j] += val_acc

    if sampling:
        return sampled_choices

    avg_val_loss = torch.mean(batch_losses_val).item()
    avg_val_acc = torch.mean(batch_accs_val).item()
    return avg_val_loss, avg_val_acc

def get_digits(string:str) -> int:
    c = ""
    nonzero = False
    for i in string:
        if i.isdigit():
            if (int(i) == 0) and (not nonzero):
                continue
            else:
                c += i
                nonzero = True
    return int(c)

def get_results_files(
                      results_dir:str,
                      modality:str,
                      version:str,
                      subfolder:str,
                      vision_model=None,
                      layer=None,
) -> list:
    if modality == 'visual':
        assert isinstance(vision_model, str) and isinstance(layer, str), 'name of vision model and layer are required'
        PATH = pjoin(results_dir, modality, vision_model, layer, version, f'{dim}d', f'{lmbda}')
    else:
        PATH = pjoin(results_dir, modality, version, f'{dim}d', f'{lmbda}')
    files = [pjoin(PATH, seed, f) for seed in os.listdir(PATH) for f in os.listdir(pjoin(PATH, seed)) if f.endswith('.json')]
    return files

def sort_results(results:dict) -> dict:
    return dict(sorted(results.items(), key=lambda kv:kv[0], reverse=False))

def merge_dicts(files:list) -> dict:
    """merge multiple .json files efficiently into a single dictionary"""
    results = {}
    for f in files:
        with open(f, 'r') as f:
            results.update(dict(json.load(f)))
    results = sort_results(results)
    return results

def load_model(
                model,
                results_dir:str,
                modality:str,
                version:str,
                data:str,
                dim:int,
                lmbda:float,
                rnd_seed:int,
                device:torch.device,
                subfolder:str='model',
):
    model_path = pjoin(results_dir, modality, version, data, f'{dim}d', f'{lmbda}', f'seed{rnd_seed:02d}', subfolder)
    models = os.listdir(model_path)
    checkpoints = list(map(get_digits, models))
    last_checkpoint = np.argmax(checkpoints)
    PATH = pjoin(model_path, models[last_checkpoint])
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_weights_(out_path:str, W_mu:torch.tensor) -> None:
    W_mu = W_mu.detach().cpu().numpy()
    W_mu = remove_zeros(W_mu)
    W_sorted = np.abs(W_mu[np.argsort(-np.linalg.norm(W_mu, ord=1, axis=1))]).T
    with open(pjoin(out_path, 'weights_sorted.npy'), 'wb') as f:
        np.save(f, W_sorted)

def load_weights(model, version:str) -> Tuple[torch.Tensor]:
    if version == 'variational':
        W_mu = model.encoder_mu[0].weight.data.T.detach()
        if hasattr(model.encoder_mu[0].bias, 'data'):
            W_mu += model.encoder_mu[0].bias.data.detach()
        W_b = model.encoder_b[0].weight.data.T.detach()
        if hasattr(model.encoder_b[0].bias, 'data'):
            W_b += model.encoder_b[0].bias.data.detach()
        W_mu = F.relu(W_mu)
        W_b = F.softplus(W_b)
        return W_mu, W_b
    else:
        return model.fc.weight.T.detach()

def prune_weights(model, version:str, indices:torch.Tensor, fraction:float):
    indices = indices[:int(len(indices)*fraction)]
    for n, m in model.named_parameters():
        if version == 'variational':
            if re.search(r'encoder', n):
                #prune output weights and biases of encoder
                m.data = m.data[indices]
            else:
                #only prune input weights of decoder
                if re.search(r'weight', n):
                    m.data = m.data[:, indices]
        else:
            #prune output weights of fc layer
            m.data = m.data[indices]
    return model

def sort_weights(model, aggregate:bool) -> np.ndarray:
    """sort latent dimensions according to their l1-norm in descending order"""
    W = load_weights(model, version='deterministic').cpu()
    l1_norms = W.norm(p=1, dim=0)
    sorted_dims = torch.argsort(l1_norms, descending=True)
    if aggregate:
        l1_sorted = l1_norms[sorted_dims]
        return sorted_dims, l1_sorted.numpy()
    return sorted_dims, W[:, sorted_dims].numpy()

def get_cut_off(klds:np.ndarray) -> int:
    klds /= klds.max(axis=0)
    cut_off = np.argmax([np.var(klds[i-1])-np.var(kld) for i, kld in enumerate(klds.T) if i > 0])
    return cut_off

def compute_kld(model, lmbda:float, aggregate:bool, reduction=None) -> np.ndarray:
    mu_hat, b_hat = load_weights(model, version='variational')
    mu = torch.zeros_like(mu_hat)
    lmbda = torch.tensor(lmbda)
    b = torch.ones_like(b_hat).mul(lmbda.pow(-1))
    kld = kld_offline(mu_hat, b_hat, mu, b)
    if aggregate:
        assert isinstance(reduction, str), '\noperator to aggregate KL divergences must be defined\n'
        if reduction == 'sum':
            #use sum as to aggregate KLDs for each dimension
            kld_sum = kld.sum(dim=0)
            sorted_dims = torch.argsort(kld_sum, descending=True)
            klds_sorted = kld_sum[sorted_dims].cpu().numpy()
        else:
            #use max to aggregate KLDs for each dimension
            kld_max = kld.max(dim=0)[0]
            sorted_dims = torch.argsort(kld_max, descending=True)
            klds_sorted = kld_max[sorted_dims].cpu().numpy()
    else:
        #use mean KLD to sort dimensions in descending order (highest KLDs first)
        sorted_dims = torch.argsort(kld.mean(dim=0), descending=True)
        klds_sorted = kld[:, sorted_dims].cpu().numpy()
    return sorted_dims, klds_sorted

#############################################################################################
######### helper functions to load weight matrices and compare RSMs across modalities #######
#############################################################################################

def load_sparse_codes(PATH) -> np.ndarray:
    Ws = [f for f in os.listdir(PATH) if f.endswith('.txt')]
    max_epoch = np.argmax(list(map(get_digits, Ws)))
    W = np.loadtxt(pjoin(PATH, Ws[max_epoch]))
    W = remove_zeros(W)
    l1_norms = np.linalg.norm(W, ord=1, axis=1)
    sorted_dims = np.argsort(l1_norms)[::-1]
    W = W[sorted_dims]
    return W.T, sorted_dims

def load_targets(model:str, layer:str, folder:str='./visual') -> np.ndarray:
    PATH = pjoin(folder, model, layer)
    with open(pjoin(PATH, 'targets.npy'), 'rb') as f:
        targets = np.load(f)
    return targets

def get_ref_indices(targets:np.ndarray) -> np.ndarray:
    n_items = len(np.unique(targets))
    cats = np.zeros(n_items, dtype=int)
    indices = np.zeros(n_items, dtype=int)
    for idx, cat in enumerate(targets):
        if cat not in cats:
            cats[cat] = cat
            indices[cat] = idx
    assert len(indices) == n_items, '\nnumber of indices for reference images must be equal to number of unique objects\n'
    return indices

def pearsonr(u:np.ndarray, v:np.ndarray, a_min:float=-1., a_max:float=1.) -> np.ndarray:
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)
    return rho

def cos_mat(W:np.ndarray, a_min:float=-1., a_max:float=1.) -> np.ndarray:
    num = matmul(W, W.T)
    l2_norms = np.linalg.norm(W, axis=1) #compute l2-norm across rows
    denom = np.outer(l2_norms, l2_norms)
    cos_mat = (num / denom).clip(min=a_min, max=a_max)
    return cos_mat

def corr_mat(W:np.ndarray, a_min:float=-1., a_max:float=1.) -> np.ndarray:
    W_c = W - W.mean(axis=1)[:, np.newaxis]
    cov = matmul(W_c, W_c.T)
    l2_norms = np.linalg.norm(W_c, axis=1) #compute l2-norm across rows
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom).clip(min=a_min, max=a_max) #counteract potential rounding errors
    return corr_mat

@njit(parallel=False, fastmath=False)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    I, K = A.shape
    K, J = B.shape
    C = np.zeros((I, J))
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                C[i, j] += A[i, k] * B[k, j]
    return C

@njit(parallel=False, fastmath=False)
def rsm_pred(W: np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    N = W.shape[0]
    S = matmul(W, W.T)
    S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
    rsm = np.zeros((N, N))
    for i in prange(N):
        for j in prange(i+1, N):
            for k in prange(N):
                if (k != i and k != j):
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])
    rsm /= N - 2
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm

def spose2rsm_odd_one_out(W: np.ndarray) -> np.ndarray:
    rsm = rsm_pred(W)
    rsm[rsm > 1] = 1
    assert np.allclose(rsm, rsm.T), '\nRSM is required to be a symmetric matrix\n'
    return rsm

def rsm(W:np.ndarray, metric:str) -> np.ndarray:
    rsm = corr_mat(W) if metric == 'rho' else cos_mat(W)
    return rsm

def compute_trils(W_mod1:np.ndarray, W_mod2:np.ndarray, metric:str) -> float:
    metrics = ['cos', 'pred', 'rho']
    assert metric in metrics, f'\nMetric must be one of {metrics}.\n'
    if metric == 'pred':
        rsm_1 = spose_rsm(W_mod1)
        rsm_2 = spose_rsm(W_mod2)
    else:
        rsm_1 = rsm(W_mod1, metric) #RSM wrt first modality (e.g., DNN)
        rsm_2 = rsm(W_mod2, metric) #RSM wrt second modality (e.g., behavior)
    assert rsm_1.shape == rsm_2.shape, '\nRSMs must be of equal size.\n'
    #since RSMs are symmetric matrices, we only need to compare their lower triangular parts (main diagonal can be omitted)
    tril_inds = np.tril_indices(len(rsm_1), k=-1)
    tril_1 = rsm_1[tril_inds]
    tril_2 = rsm_2[tril_inds]
    return tril_1, tril_2, tril_inds

def compare_modalities(W_mod1:np.ndarray, W_mod2:np.ndarray, duplicates:bool=False) -> Tuple[np.ndarray]:
    assert W_mod1.shape[0] == W_mod2.shape[0], '\nNumber of items in weight matrices must align.\n'
    mod1_mod2_corrs = np.zeros(W_mod1.shape[1])
    mod2_dims = []
    for d_mod1, w_mod1 in enumerate(W_mod1.T):
        corrs = np.array([pearsonr(w_mod1, w_mod2) for w_mod2 in W_mod2.T])
        if duplicates:
            mod2_dims.append(np.argmax(corrs))
        else:
            for d_mod2 in np.argsort(-corrs):
                if d_mod2 not in mod2_dims:
                    mod2_dims.append(d_mod2)
                    break
        mod1_mod2_corrs[d_mod1] = corrs[mod2_dims[-1]]
    mod1_dims_sorted = np.argsort(-mod1_mod2_corrs)
    mod2_dims_sorted = np.asarray(mod2_dims)[mod1_dims_sorted]
    corrs = mod1_mod2_corrs[mod1_dims_sorted]
    return mod1_dims_sorted, mod2_dims_sorted, corrs

def sparsity(A:np.ndarray) -> float:
    return 1.0 - (A[A>0].size/A.size)

def avg_sparsity(Ws:list) -> np.ndarray:
    return np.mean(list(map(sparsity, Ws)))

def robustness(corrs:np.ndarray, thresh:float) -> float:
    return len(corrs[corrs>thresh])/len(corrs)

def cross_correlate_latent_dims(X, thresh:float=None) -> float:
    if isinstance(X, np.ndarray):
        W_mu_i = np.copy(X)
        W_mu_j = np.copy(X)
    else:
        W_mu_i, W_mu_j = X
    corrs = np.zeros(min(W_mu_i.shape))
    for i, w_i in enumerate(W_mu_i):
        if np.all(W_mu_i == W_mu_j):
            corrs[i] = np.max([pearsonr(w_i, w_j) for j, w_j in enumerate(W_mu_j) if j != i])
        else:
            corrs[i] = np.max([pearsonr(w_i, w_j) for w_j in W_mu_j])
    if thresh:
        return robustness(corrs, thresh)
    return np.mean(corrs)
