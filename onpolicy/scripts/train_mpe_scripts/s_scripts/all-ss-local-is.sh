#!/bin/bash
script_name="train_mpe_ss_local_is.sh"
seed=0


sbatch -n 1 --cpus-per-task=2 --gpus=1 --gres=gpumem:2048m --time=18:00:00 --mem-per-cpu=12288 --wrap="./call_experiment.sh ${script_name} ${seed} 12 0.0"
sbatch -n 1 --cpus-per-task=2 --gpus=1 --gres=gpumem:2048m --time=18:00:00 --mem-per-cpu=12288 --wrap="./call_experiment.sh ${script_name} ${seed} 12 0.1"
sbatch -n 1 --cpus-per-task=2 --gpus=1 --gres=gpumem:2048m --time=18:00:00 --mem-per-cpu=12288 --wrap="./call_experiment.sh ${script_name} ${seed} 12 0.01"

sbatch -n 1 --cpus-per-task=2 --gpus=1 --gres=gpumem:2048m --time=18:00:00 --mem-per-cpu=12288 --wrap="./call_experiment.sh ${script_name} ${seed} 26 0.0"
sbatch -n 1 --cpus-per-task=2 --gpus=1 --gres=gpumem:2048m --time=18:00:00 --mem-per-cpu=12288 --wrap="./call_experiment.sh ${script_name} ${seed} 26 0.1"
sbatch -n 1 --cpus-per-task=2 --gpus=1 --gres=gpumem:2048m --time=18:00:00 --mem-per-cpu=12288 --wrap="./call_experiment.sh ${script_name} ${seed} 26 0.01"



