#!/bin/bash
script_name="train_mpe_ss_local_ip.sh"
seed=0

sbatch --wrap="./ss-ip.sh ${script_name} ${seed} 12 0.0"
sbatch --wrap="./ss-ip.sh ${script_name} ${seed} 12 0.1"
sbatch --wrap="./ss-ip.sh ${script_name} ${seed} 12 0.01"
sbatch --wrap="./ss-ip.sh ${script_name} ${seed} 26 0.0"
sbatch --wrap="./ss-ip.sh ${script_name} ${seed} 26 0.1"
sbatch --wrap="./ss-ip.sh ${script_name} ${seed} 26 0.01"



