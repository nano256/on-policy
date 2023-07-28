#!/bin/bash

# IP, CI=26, CC=0.01
source ./train_mpe_ss_local_ip.sh 0 26 0.01
source ./train_mpe_ss_local_ip.sh 1 26 0.01
source ./train_mpe_ss_local_ip.sh 2 26 0.01
source ./train_mpe_ss_local_ip.sh 3 26 0.01
source ./train_mpe_ss_local_ip.sh 4 26 0.01

# IP, CI=26, CC=0.01
source ./train_mpe_ss_local_ip.sh 0 26 0.1
source ./train_mpe_ss_local_ip.sh 1 26 0.1
source ./train_mpe_ss_local_ip.sh 2 26 0.1
source ./train_mpe_ss_local_ip.sh 3 26 0.1
source ./train_mpe_ss_local_ip.sh 4 26 0.1

# IP, CI=26, CC=1.0
source ./train_mpe_ss_local_ip.sh 0 26 1.0
source ./train_mpe_ss_local_ip.sh 1 26 1.0
source ./train_mpe_ss_local_ip.sh 2 26 1.0
source ./train_mpe_ss_local_ip.sh 3 26 1.0
source ./train_mpe_ss_local_ip.sh 4 26 1.0