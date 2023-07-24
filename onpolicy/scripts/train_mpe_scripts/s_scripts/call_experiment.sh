#!/bin/bash

script_name=$1
seed=$2
communication_interval=$3
commitment_coef=$4

module load gcc/8.2.0
module load python_gpu/3.11.2

source /cluster/home/miperez/venvs/onpolicy311/bin/activate


source "../experiments/${script_name}" ${seed} ${communication_interval} ${commitment_coef}
