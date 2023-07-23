#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:2048m
#SBATCH --time=6:00:00
#SBATCH --job-name="onpolicy_core_test"
#SBATCH --mem-per-cpu=3072
#SBATCH --mail-type=END
#SBATCH --mail-user=miperez@ethz.ch

source ../experiments/train_mpe_ss_local_ip.sh 0 12 0.01
