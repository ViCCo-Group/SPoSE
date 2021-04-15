#!/bin/bash -l

#SBATCH -a 0-19
#SBATCH -o ./job_%A_%a.out
#SBATCH -e ./job_%A_%a.err
#SBATCH -D ./
#SBATCH -J spose_behavioral_data
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=38000M
#SBATCH --mail-type=none
#SBATCH --mail-user=lmutt@rzg.mpg.de

module purge
module load gcc/8
module load impi/2019.9
module load anaconda/3/2020.02
module load pytorch/cpu/1.7.0

export OMP_NUM_THREADS=1

TASK='odd_one_out'
MODALITY='behavioral/'
TR_DIR='./triplets/behavioral/'
LR="0.001"
DIM=100
T=1000
BS=128
WS=900
DEVICE='cpu'
RND_SEEDS=(0 1 2 3 4 5 6 7 8 9 10 21 22 23 24 25 26 27 29 42) 

echo "Started SPoSE $DIM optimization at $(date)"

srun python3 ./train.py --task $TASK --modality $MODALITY --triplets_dir $TR_DIR --learning_rate $LR --embed_dim $DIM --batch_size $BS --epochs $T --n_models $SLURM_CPUS_PER_TASK --window_size $WS --plot_dims --device $DEVICE --rnd_seed ${RND_SEEDS[$SLURM_ARRAY_TASK_ID]}  >> spose_behavioral_parallel.out

echo "Finished SPoSE $DIM optimization at $(date)"