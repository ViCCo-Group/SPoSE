#!/bin/bash -l
#SBATCH -a 0-16
#SBATCH -o ./job_%A_%a.out
#SBATCH -e ./job_%A_%a.err
#SBATCH -D ./
#SBATCH -J spose_behavioral_data
#SBATCH --time=24:00:00

#SBATCH --partition="small"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=76000M
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
T=1000
BS=128
WS=400
DEVICE='cpu'
TR_DIRS=("./triplets/behavioral/10/split_01" "./triplets/behavioral/10/split_02" "./triplets/behavioral/10/split_03" "./triplets/behavioral/10/split_04" "./triplets/behavioral/10/split_05" "./triplets/behavioral/10/split_06" "./triplets/behavioral/10/split_07" "./triplets/behavioral/10/split_08" "./triplets/behavioral/10/split_09" "./triplets/behavioral/10/split_10" "./triplets/behavioral/20/split_01" "./triplets/behavioral/20/split_02" "./triplets/behavioral/20/split_03" "./triplets/behavioral/20/split_04" "./triplets/behavioral/20/split_05" "./triplets/behavioral/50/split_01" "./triplets/behavioral/50/split_02")
LAMBDAS=("0.008" "0.0081" "0.0082" "0.0083" "0.0084" "0.0085" "0.0086" "0.0087" "0.0088" "0.0089" "0.009" "0.0091" "0.0092" "0.0093" "0.0094" "0.0095" "0.0096" "0.0097" "0.0098" "0.0099")
RND_SEED=18

echo "Started SPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"

srun python3 ./train.py --task $TASK --modality $MODALITY --triplets_dir ${TR_DIRS[$SLURM_ARRAY_TASK_ID]} --learning_rate $LR --embed_dim $DIM --batch_size $BS --epochs $T --n_models ${LAMBDAS[@]} --window_size $WS --plot_dims --device $DEVICE --rnd_seed $RND_SEED  >> spose_behavioral_parallel.out

echo "Finished SPoSE $SLURM_ARRAY_TASK_ID optimization at $(date)"
