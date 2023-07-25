import numpy as np

import gym
import gymnasium
from gym import spaces
from pettingzoo.sisl import pursuit_v4

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


class TransposeObsWrapper(gym.Env):
    def __init__(self, env, obs_transpose):
        super().__init__()
        self.env = env
        self._obs_transpose = obs_transpose

        self.observation_space = self._arrange_obs_spaces(self.env.observation_space)

    def reset(self):
        obs = self.env.reset()
        obs = self._arrange_obs(obs)
        return obs

    def step(self, actions):
        obs, rewards, terminated, info = self.env.step(actions)
        obs = self._arrange_obs(obs)
        return obs, rewards, terminated, info

    def _arrange_obs(self, obs):
        # Env gives following order of dims: (height, width, channels)
        # We want following order of dims: (channels, height, width)
        new_obs = []
        for o in obs:
            new_obs.append(np.transpose(o, self._obs_transpose))
        return new_obs

    def _arrange_obs_spaces(self, obs_spaces):
        if isinstance(obs_spaces, list):
            new_obs_spaces = []
            for os in obs_spaces:
                new_obs_spaces.append(self._arrange_obs_space(os))
            return new_obs_spaces
        else:
            return self._arrange_obs_space(obs_spaces)

    def _arrange_obs_space(self, obs_space):
        if isinstance(obs_space, (spaces.Box, gymnasium.spaces.Box)):
            upper_bounds = np.transpose(obs_space.high, self._obs_transpose)
            lower_bounds = np.transpose(obs_space.low, self._obs_transpose)
            return spaces.Box(lower_bounds, upper_bounds)
        else:
            raise NotImplementedError("This space is not supported yet.")

    def __getattr__(self, attr):
        return getattr(self.env, attr)


class PettingZooToOnPolicyWrapper(gym.Env):
    def __init__(self, env, seed):
        super().__init__()
        self.env = env
        self.seed = seed
        self.n = self.env.max_num_agents
        obs_shape = self.env.observation_space(self.env.possible_agents[0]).shape
        self.observation_space = [
            self.env.observation_space(agent) for agent in self.env.possible_agents
        ]
        self.action_space = [
            self.env.action_space(agent) for agent in self.env.possible_agents
        ]

        self.share_observation_space = [
            spaces.Box(-np.inf, np.inf, (int(np.product(obs_shape)) * self.n,))
            for _ in self.env.possible_agents
        ]

    def reset(self):
        obs, _ = self.env.reset(self.seed)
        obs = self._dict_to_array(obs)
        return obs

    def step(self, actions):
        actions = self._convert_actions(actions)
        actions = self._array_to_dict(actions)
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        if np.any(list(terminated.values())):
            print("done")

        obs = self._dict_to_array(obs)
        rewards = self._dict_to_array(rewards)
        rewards = [[r] for r in rewards]
        terminated = self._dict_to_array(terminated)
        truncated = self._dict_to_array(truncated)
        terminated = [a or b for a, b in zip(terminated, truncated)]
        info = self._dict_to_array(info)
        return obs, rewards, terminated, info

    def _convert_actions(self, actions):
        converted_actions = []
        for a in actions:
            converted_actions.append(np.argmax(a).item())
        return converted_actions

    def _dict_to_array(self, d):
        a = []
        for agent in self.env.possible_agents:
            a.append(d[agent])
        return a

    def _array_to_dict(self, a):
        d = {}
        for agent, value in zip(self.env.possible_agents, a):
            d[agent] = value
        return d

    def __getattr__(self, attr):
        return getattr(self.env, attr)


def SISLEnv(args, seed):
    if args.scenario_name == "pursuit":
        return TransposeObsWrapper(
            PettingZooToOnPolicyWrapper(
                pursuit_v4.parallel_env(
                    max_cycles=args.episode_length, n_pursuers=args.num_agents
                ),
                seed,
            ),
            (2, 0, 1),
        )
    else:
        NotImplementedError(f"{args.scenario_name} is not implemented in SISL.")
