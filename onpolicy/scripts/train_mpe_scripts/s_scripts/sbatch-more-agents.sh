#!/bin/bash

is_script_name="train_mpe_ss_local_is.sh"
ip_script_name="train_mpe_ss_local_ip.sh"



sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 26 0.0 5 5"
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 26 0.0 3 5"

sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 0 26 0.01 5 5"
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 0 26 0.01 3 5"
