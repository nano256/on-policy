import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class MPECommunicationRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(MPECommunicationRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    messages,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                    trajectories,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    messages,
                    rnn_states,
                    rnn_states_critic,
                    trajectories,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if "individual_reward" in info[agent_id].keys():
                                idv_rews.append(info[agent_id]["individual_reward"])
                        agent_k = "agent%i/individual_rewards" % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = (
                    np.mean(self.buffer.rewards) * self.episode_length
                )
                print(
                    "average episode rewards is {}".format(
                        train_infos["average_episode_rewards"]
                    )
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        if self.all_args.model == "is" and self.all_args.pretrain_world_model:
            self.pretrain_world_model(
                self.trainer.policy.actor,
                self.envs,
                self.num_agents,
                self.all_args.pretrain_wm_n_samples,
                self.all_args.pretrain_wm_batch_size,
                self.all_args.pretrain_wm_n_episodes,
            )
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            message,
            rnn_states,
            rnn_states_critic,
            trajectory,
        ) = self.trainer.policy.get_actions(
            self.buffer.share_obs[step],
            self.buffer.obs[step],
            self.buffer.messages[step],
            self.buffer.rnn_states[step],
            self.buffer.rnn_states_critic[step],
            self.buffer.masks[step],
            step,
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        messages = np.array(np.split(_t2n(message), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        if trajectory is not None:
            trajectories = np.array(np.split(_t2n(trajectory), self.n_rollout_threads))
        else:
            trajectories = None
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[
                    actions[:, :, i]
                ]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            messages,
            rnn_states,
            rnn_states_critic,
            actions_env,
            trajectories,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            messages,
            rnn_states,
            rnn_states_critic,
            trajectories,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            messages=messages,
            trajectories=trajectories,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] + 1
                    )[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2
                        )
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2
                )
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                            actions[:, :, i]
                        ]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate(
                                (actions_env, uc_actions_env), axis=2
                            )
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print(
                "average episode rewards is: "
                + str(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            )

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )

    def pretrain_world_model(
        self, model, env, n_agents, n_samples, batch_size, n_episodes
    ):
        # Since each env step gives the number of agents in samples
        samples_per_env_step = n_agents * env.num_envs
        obs_batch = []
        action_batch = []
        # env reset only gives back the first observation
        obs = env.reset()
        obs_rollout = [obs]
        action_rollout = []
        n_actions = env.action_space[0].n
        actions_per_step = np.product(obs.shape[:-1])
        obs_predictor = model.observation_predictor
        collected_samples = 0
        # Collect observations from random trajectories
        while True:
            random_actions = torch.randint(n_actions, (actions_per_step,)).reshape(
                (*obs.shape[:-1], 1)
            )
            random_actions = F.one_hot(random_actions, n_actions).squeeze()
            obs, _, dones, _ = env.step(random_actions)
            obs_rollout.append(obs)
            action_rollout.append(random_actions.numpy())
            if np.any(dones):
                obs_batch.append(obs_rollout)
                action_batch.append(action_rollout)
                collected_samples += len(action_rollout) * samples_per_env_step
                if collected_samples >= n_samples:
                    break
                obs_rollout = [env.reset()]
                action_rollout = []

        obs_batch = torch.Tensor(np.array(obs_batch))
        action_batch = torch.Tensor(np.array(action_batch))
        action_batch = self._preprocess_actions(action_batch)
        # Take all obs except the last of each episode as the input data
        initial_obs = obs_batch[:, :-1, ...]
        x_train = torch.cat((initial_obs, action_batch), -1)
        x_train = x_train.reshape(-1, x_train.size(-1))[:n_samples]
        # Take all consecutive obs of x as our labels
        y_train = obs_batch[:, 1:, ...].reshape(-1, obs_batch.size(-1))[:n_samples]
        initial_obs_train = initial_obs.reshape(-1, obs_batch.size(-1))[:n_samples]

        n_batches = n_samples // batch_size

        optim = torch.optim.Adam(obs_predictor.parameters())
        loss_fn = nn.MSELoss()
        steps = 0
        for _ in range(n_episodes):
            samples = torch.randperm(n_samples)
            for n_batch in range(n_batches):
                optim.zero_grad()
                idx = samples[batch_size * n_batch : batch_size * (n_batch + 1)]
                x = x_train[idx]
                x_obs = initial_obs_train[idx]
                y = y_train[idx]
                # The observation predictor infers the the difference between current and next observation. Therefore we add the current obs to the output.
                y_pred = obs_predictor(x) + x_obs
                loss = loss_fn(y_pred, y)
                loss.backward()
                optim.step()
                # TODO: Replace with proper tensorboard log
                print(f"WM obs. predictor loss at step {steps}: {loss:.4f}")
                self.log_train({"wm_obs_pred_pretrain_loss": loss.item()}, steps)
                steps += batch_size

    def _preprocess_actions(self, actions):
        if actions.dim() == 5:
            # We assume actions having the dims (Rollout, Sequence, Batch, Agent, Features)
            action_shape = actions.shape
            n_agents = actions.size(3)
            own_action = []
            other_actions = []
            for idx in range(n_agents):
                mask = torch.ones_like(actions)
                mask[..., idx, :] = 0
                mask = mask == 1
                other_actions.append(actions[mask].reshape(*action_shape[:3], 1, -1))
                # We want to keep the agent dim
                own_action.append(actions[..., idx : idx + 1, :])
            other_actions = torch.cat(other_actions, 3)
            own_action = torch.cat(own_action, 3)
            all_actions = torch.cat((own_action, other_actions), -1)
        elif actions.dim() == 4:
            # We assume actions having the dims (Seq, Batch, Agent, Features)
            action_shape = actions.shape
            n_agents = actions.shape[2]
            own_action = []
            other_actions = []
            for idx in range(n_agents):
                mask = torch.ones_like(actions)
                mask[:, :, idx, :] = 0
                mask = mask == 1
                other_actions.append(actions[mask].reshape(*action_shape[:2], 1, -1))
                # We want to keep the agent dim
                own_action.append(actions[..., idx : idx + 1, :])
            other_actions = torch.cat(other_actions, 2)
            own_action = torch.cat(own_action, 2)
            all_actions = torch.cat((own_action, other_actions), -1)
        else:
            ValueError(
                f"Input expected to have 4 or 5 dimensions, got {actions.dim()} instead."
            )
        return all_actions
