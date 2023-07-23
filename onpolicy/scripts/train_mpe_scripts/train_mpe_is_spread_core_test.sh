#!/bin/sh
env="MPE"
scenario="simple_spread" 
exp_prefix="test"
num_landmarks=3
num_agents=3
episode_length=25
user_name="miperez"
seed_max=1

num_env_steps=100000
lr=7e-4
critic_lr=7e-4
gain=0.01
ppo_epoch=10
n_rollout_threads=128
n_training_threads=1
num_mini_batch=1

pretrain_wm_n_samples=1000
pretrain_wm_batch_size=500
pretrain_wm_n_episodes=10

imagined_traj_len=4
communication_interval=4

echo "Current experiment: CRMAPPO, plain IS"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --user_name ${user_name} --use_wandb 0 \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} \
    --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} \
    --ppo_epoch ${ppo_epoch} --use_ReLU --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --wandb_name ${exp} \
    --imagined_traj_len ${imagined_traj_len} --communication_interval 1 --pretrain_world_model 1 --pretrain_wm_n_samples ${pretrain_wm_n_samples} \
    --pretrain_wm_batch_size ${pretrain_wm_batch_size} --pretrain_wm_n_episodes ${pretrain_wm_n_episodes}  \
     --model "is" --algorithm_name "crmappo" --experiment_name "${exp_prefix}_is_plain"
done