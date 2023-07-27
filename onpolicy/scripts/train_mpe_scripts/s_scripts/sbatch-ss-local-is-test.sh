#!/bin/bash

is_script_name="train_mpe_ss_local_is.sh"
ip_script_name="train_mpe_ss_local_ip.sh"

sbatch --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --time=48:00:00 --mem-per-cpu=4096 --wrap="./call_experiment.sh ${is_script_name} 0 26 0.0"
sbatch --cpus-per-task=8 --time=48:00:00 --mem-per-cpu=4096 --wrap="./call_experiment.sh ${is_script_name} 1 26 0.0" 
            
