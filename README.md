#### Environment Setup

1. The code uses Python 3.8,  [Pytorch 1.6.0](https://pytorch.org/) (Note that PyTorch 1.6.0 requires CUDA 10.2, if you want to run on a GPU)
2. Install PyTorch: `pip install pytorch` or `conda install pytorch torchvision -c pytorch` (the latter is recommended if you use Anaconda)
3. Install Python dependencies: `pip install -r requirements.txt`

#### Train SPoSE model 

```
  python train.py
  
 --task (specify whether you'd like the model to perform an odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task)
 --modality (define for which modality specified task should be performed by SPoSE (e.g., behavioral, text, visual))
 --triplets_dir (in case you have tripletized data, provide directory from where to load triplets)
 --results_dir (optional specification of results directory (if not provided will resort to ./results/modality/version/dim/lambda/seed/))
 --plots_dir (optional specification of directory for plots (if not provided will resort to ./plots/modality/version/dim/lambda/seed/)
 --learning_rate (learning rate to be used in optimizer)
 --lmbda (lambda value determines l1-norm fraction to regularize loss; will be divided by number of items in the original data matrix)
 --embed_dim (embedding dimensionality, i.e., output size of the neural network)
 --batch_size (batch size)
 --epochs (maximum number of epochs to optimize SPoSE model for)
 --window_size (window size to be used for checking convergence criterion with linear regression)
 --sampling_method (sampling method; if soft, then you can specify a fraction of your training data to be sampled from during each epoch; else full train set will be used)
 --steps (save model parameters and create checkpoints every <steps> epochs)
 --resume (bool) (whether to resume training at last checkpoint; if not set training will restart)
 --p (fraction of train set to sample; only necessary for *soft* sampling)
 --device (CPU or CUDA)
 --rnd_seed (random seed)
 --early_stopping (bool) (train until convergence)
 --num_threads (number of threads used by PyTorch multiprocessing)
```

Here is an example call for single-process training:

```
python train.py --task odd_one_out --modality behavioral/ --triplets_dir ./triplets/behavioral/ --learning_rate 0.001 --lmbda 0.008 --embed_dim 100 --batch_size 128 --epochs 500 --window_size 50 --steps 5 --sampling_method normal --device cuda --rnd_seed 42
```

#### NOTES:

1. Note that the triplets are expected to be in the format `N x 3`, where N = number of trials (e.g., 100k) and 3 refers to the triplets, where `col_0` = anchor_1, `col_1` = anchor_2, `col_2` = odd one out. Triplet data must be split into train and test splits, and named `train_90.txt` and `test_10.txt` respectively. In case you would like to use some sort of text embeddings (e.g., sensvecs), simply put your `.csv` files into a folder that refers to the current modality (e.g., `./text/`), and the script will automatically tripletize the word embeddings for you and move the triplet data into `./triplets/text/`.

2. The script automatically saves the weight matrix `W` of the SPoSE model at each convergence checkpoint. 

3. The script plots train and test performances alongside each other for each lambda value. All plots can be found in `./plots/` after model convergence.

4. For a specified lambda value, you get a `.json` file where both the best test performance(s) and the corresponding epoch at `max` performance are stored. You find the file in the results folder.

5. The number of non-negative dimensions (i.e., weights > 0.1) gets plotted as a function of time after the model has converged. This is useful to qualitatively inspect changes in non-negative dimensions over training epochs. Again, plots can be found in `./plots/` after model convergence.
