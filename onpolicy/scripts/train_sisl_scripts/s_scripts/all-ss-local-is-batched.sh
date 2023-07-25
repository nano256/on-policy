#!/bin/bash
is_script_name="train_sisl_ss_local_is.sh"
seed=0


sbatch -n 3 --cpus-per-task=6 --gpus=1 --gres=gpumem:24576m --time=48:00:00 --mem-per-cpu=4096 \
    --wrap="./call_experiment.sh ${is_script_name} ${seed} 12 0.0 & \
            ./call_experiment.sh ${is_script_name} ${seed} 12 0.1 & \
            ./call_experiment.sh ${is_script_name} ${seed} 12 0.01 "



