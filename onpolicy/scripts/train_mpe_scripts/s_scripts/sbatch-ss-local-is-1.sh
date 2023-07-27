#!/bin/bash

#SBATCH --ntasks 3
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --gres=gpumem:6000m
#SBATCH --time=48:00:00
#SBATCH --job-name="batch run 1"
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=miperez@ethz.ch

is_script_name="train_sisl_ss_local_is.sh"
ip_script_name="train_sisl_ss_local_ip.sh"

srun --ntasks=1 --exclusive --cpus-per-task=6 --gpus=1 --gres=gpumem:2000m --time=48:00:00 --mem-per-cpu=4096 \
    --wrap="./call_experiment.sh ${ip_script_name} 1 26 0.0" &
srun --ntasks=1 --exclusive --cpus-per-task=6 --gpus=1 --gres=gpumem:2000m --time=48:00:00 --mem-per-cpu=4096 \
    --wrap="./call_experiment.sh ${ip_script_name} 2 26 0.0" &
srun --ntasks=1 --exclusive --cpus-per-task=6 --gpus=1 --gres=gpumem:2000m --time=48:00:00 --mem-per-cpu=4096 \
    --wrap="./call_experiment.sh ${ip_script_name} 3 26 0.0" &
wait
            
