#!/bin/bash

is_script_name="train_sisl_ss_local_is.sh"
ip_script_name="train_sisl_ss_local_ip.sh"

sbatch --ntasks=1 --exclusive --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --time=48:00:00 --mem-per-cpu=4096 \
    --wrap="./call_experiment.sh ${is_script_name} 0 26 0.0"
sbatch --ntasks=1 --exclusive --cpus-per-task=8 --time=48:00:00 --mem-per-cpu=4096 \
    --wrap="./call_experiment.sh ${is_script_name} 1 26 0.0" 
            
