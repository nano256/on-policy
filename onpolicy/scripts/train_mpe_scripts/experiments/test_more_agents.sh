#!/bin/bash

# IS, CI=26, CC=0.01, NA=5, NL=5
source ./train_mpe_ss_local_is.sh 0 26 0.01 5 5
# IS, CI=26, CC=0.01, NA=3, NL=5
source ./train_mpe_ss_local_is.sh 0 26 0.01 3 5

# IP, CI=26, CC=0.01, NA=5, NL=5
source ./train_mpe_ss_local_ip.sh 0 26 0.01 5 5
# IP, CI=26, CC=0.01, NA=3, NL=5
source ./train_mpe_ss_local_ip.sh 0 26 0.01 3 5
