#!/bin/bash
is_script_name="train_mpe_ss_local_is.sh"
ip_script_name="train_mpe_ss_local_ip.sh"
seed=0


sbatch -n 12 --cpus-per-task=2 --gpus=1 --gres=gpumem:24576m --time=48:00:00 --mem-per-cpu=12288 \
    --wrap="./call_experiment.sh ${is_script_name} ${seed} 12 0.0 & \
            ./call_experiment.sh ${is_script_name} ${seed} 12 0.1 & \
            ./call_experiment.sh ${is_script_name} ${seed} 12 0.01 & \
            ./call_experiment.sh ${is_script_name} ${seed} 26 0.0 & \
            ./call_experiment.sh ${is_script_name} ${seed} 26 0.1 & \
            ./call_experiment.sh ${is_script_name} ${seed} 26 0.01 & \
            ./call_experiment.sh ${ip_script_name} ${seed} 12 0.0 & \
            ./call_experiment.sh ${ip_script_name} ${seed} 12 0.1 & \
            ./call_experiment.sh ${ip_script_name} ${seed} 12 0.01 & \
            ./call_experiment.sh ${ip_script_name} ${seed} 26 0.0 & \
            ./call_experiment.sh ${ip_script_name} ${seed} 26 0.1 & \
            ./call_experiment.sh ${ip_script_name} ${seed} 26 0.01 "



