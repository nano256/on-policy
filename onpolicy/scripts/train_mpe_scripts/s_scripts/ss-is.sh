#!/bin/bash

script_name=$1
seed=$2
communication_interval=$3
commitment_coef=$4

#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gres=gpumem:2048m
#SBATCH --time=6:00:00
#SBATCH --job-name="ss_ip"
#SBATCH --mem-per-cpu=12000
#SBATCH --mail-type=END
#SBATCH --mail-user=miperez@ethz.ch

module load gcc/8.2.0
module load python_gpu/3.11.2

source /cluster/home/miperez/venvs/onpolicy311/bin/activate


source "../experiments/${script_name}" ${seed} ${communication_interval} ${commitment_coef}
