#!/bin/bash -l

#SBATCH -a 0-9
#SBATCH -o ./job_%A_%a.out
#SBATCH -e ./job_%A_%a.err
#SBATCH -D ./
#SBATCH -J spose_behavioral_data
#SBATCH --time=24:00:00
#SBATCH --nodes=1
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
MODALITY='behavioral'
LR="0.001"
DIM=100
T=500
BS=128
WS=400
DEVICE='cpu'
TR_DIRS=("./triplets/behavioral/split_01" "./triplets/behavioral/split_02" "./triplets/behavioral/split_03" "./triplets/behavioral/split_04" "./triplets/behavioral/split_05" "./triplets/behavioral/split_06" "./triplets/behavioral/split_07" "./triplets/behavioral/split_08" "./triplets/behavioral/split_09" "./triplets/behavioral/split_10")
RND_SEED=42

echo "Started SPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"

srun python3 ./train.py --task $TASK --modality $MODALITY --triplets_dir ${TR_DIRS[$SLURM_ARRAY_TASK_ID]} --learning_rate $LR --embed_dim $DIM --batch_size $BS --epochs $T --n_models $SLURM_CPUS_PER_TASK --window_size $WS --plot_dims --device $DEVICE --rnd_seed $RND_SEED  >> spose_behavioral_parallel.out

echo "Finished SPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"
