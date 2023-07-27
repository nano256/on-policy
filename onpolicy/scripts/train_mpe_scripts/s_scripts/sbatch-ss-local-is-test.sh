#!/bin/bash

is_script_name="train_mpe_ss_local_is.sh"
ip_script_name="train_mpe_ss_local_ip.sh"

sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 26 0.0"
sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --wrap="./call_experiment.sh ${is_script_name} 1 26 0.0" 
            
sbatch -n 1 --cpus-per-task=18 --gpus=1 --gres=gpumem:6144m --time=48:00:00 --job-name="batch run 1" --mem-per-cpu=4096  --wrap="echo hello"