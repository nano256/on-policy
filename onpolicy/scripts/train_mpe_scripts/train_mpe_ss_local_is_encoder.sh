#!/bin/sh

seed=${1-0}
imagined_traj_len=${2-1}
communication_interval=${2-1}
commitment_coef=${3-0.0}
num_agents=${4-3}
num_landmarks=${5-$num_agents}

env="MPE"
scenario="simple_spread" 
exp_prefix="IS_IPPO_local"
episode_length=25
user_name="miperez"

num_env_steps=10000000
lr=7e-4
critic_lr=7e-4
gain=0.01
ppo_epoch=10
n_rollout_threads=128
n_training_threads=1
num_mini_batch=1

pretrain_wm_n_samples=10000
pretrain_wm_batch_size=500
pretrain_wm_n_episodes=10

intention_aggregation="encoder"


echo "env is ${env}, scenario is ${scenario}, seed is ${seed}"

echo "Current experiment: CRIPPO, IS, local obs only"

echo "communication_interval: ${communication_interval}, commitment_coef: ${commitment_coef}, num_agents: ${num_agents}, num_landmarks: ${num_landmarks}"

CUDA_VISIBLE_DEVICES=0 python ../../train/train_mpe.py --env_name ${env} --user_name ${user_name} --use_wandb 0 \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} \
    --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} --num_env_steps ${num_env_steps} \
    --ppo_epoch ${ppo_epoch} --use_ReLU --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --wandb_name ${exp} \
    --pretrain_world_model 1 --pretrain_wm_n_samples ${pretrain_wm_n_samples} \
    --pretrain_wm_batch_size ${pretrain_wm_batch_size} --pretrain_wm_n_episodes ${pretrain_wm_n_episodes} \
    --model "is" --algorithm_name "crmappo" --experiment_name "${exp_prefix}_seed_${seed}_comm_int_${communication_interval}_commit_coef_${commitment_coef}_n_agents_${num_agents}_n_landmarks_${num_landmarks}_ia_${intention_aggregation}" --use_commitment_loss 1 \
    --imagined_traj_len ${imagined_traj_len} --communication_interval ${communication_interval} --use_centralized_V 0 \
    --use_local_obs 1 --commitment_coef ${commitment_coef} --intention_aggregation ${intention_aggregation} --use_prob_dist_traj 1


