#!/bin/bash

is_script_name="train_mpe_ss_local_is.sh"
ip_script_name="train_mpe_ss_local_ip.sh"


# IP, CI=26, CC=0.01
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 0 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 1 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 2 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 3 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 4 26 0.01"

# IP, CI=26, CC=0.1
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 0 26 0.1"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 1 26 0.1"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 2 26 0.1"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 3 26 0.1"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 4 26 0.1"

# IP, CI=26, CC=1.0
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 0 26 1.0"
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 1 26 1.0"
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 2 26 1.0"
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 3 26 1.0"
sbatch -n 1 --cpus-per-task=20 --gpus=1 --gres=gpumem:3000m --job-name="IP GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${ip_script_name} 4 26 1.0"

# IS, CI=26, CC=0.01
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 1 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 2 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 3 26 0.01"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 4 26 0.01"

# # IS, CI=1, CC=0.0
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 1 0.0"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 1 1 0.0"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 2 1 0.0"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 3 1 0.0"
# sbatch -n 1 --cpus-per-task=8 --gpus=1 --gres=gpumem:2000m --job-name="IS GPU" --time=48:00:00 --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 4 1 0.0"

# # IS, CI=12, CC=0.0
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 12 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 1 12 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 2 12 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 3 12 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 4 12 0.0" 

# # IS, CI=26, CC=0.0
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 0 26 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 1 26 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 2 26 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 3 26 0.0" 
# sbatch -n 1 --cpus-per-task=8 --time=48:00:00 --job-name="IS CPU" --mem-per-cpu=4096 --mail-type=BEGIN,END --mail-user=miperez@ethz.ch --wrap="./call_experiment.sh ${is_script_name} 4 26 0.0" 