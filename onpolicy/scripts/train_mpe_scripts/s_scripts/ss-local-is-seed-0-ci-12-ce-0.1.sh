#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:2048m
#SBATCH --time=6:00:00
#SBATCH --job-name="ss-local-is"
#SBATCH --mem-per-cpu=3072
#SBATCH --mail-type=END
#SBATCH --mail-user=miperez@ethz.ch

module load gcc/8.2.0
module load python_gpu/3.11.2

source /cluster/home/miperez/venvs/onpolicy311/bin/activate

source ../experiments/train_mpe_ss_local_is.sh 0 12 0.1
