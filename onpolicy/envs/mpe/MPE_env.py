import numpy as np
from gym import spaces
from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    if args.scenario_name == "simple_spread" and args.use_local_obs:
        env = SimpleSpreadLocalObsEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.info,
        )
    else:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.info,
        )

    return env


class SimpleSpreadLocalObsEnv(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        The obs of an agent in simple spread are 
        [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
        In the source code of the env, we see that self_vel.dim() == self_pos.dim() == 2 and
        landmark_rel_positions.dim() == other_agent_rel_positions.dim() == communication.dim()
        == n_agents * 2. Since communication in our use case is not used, we can cut away
        the last n_agents * 4 values of the agent's obs to restrict it to local observations.
        """
        self.cut_obs = (self.n - 1) * 4
        new_obs_dim = self.observation_space[0].shape[0] - self.cut_obs
        new_shared_obs_dim = new_obs_dim * self.n
        self.observation_space = [
            spaces.Box(-np.inf, np.inf, (new_obs_dim,)) for _ in range(self.n)
        ]
        self.share_observation_space = [
            spaces.Box(-np.inf, np.inf, (new_shared_obs_dim,)) for _ in range(self.n)
        ]

    def step(self, action_n):
        obs_n, reward_n, done_n, info_n = super().step(action_n)
        obs_n = self._cut_obs(obs_n)
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = super().reset()
        obs_n = self._cut_obs(obs_n)
        return obs_n

    def _cut_obs(self, obs_n):
        return [obs[: -self.cut_obs] for obs in obs_n]
