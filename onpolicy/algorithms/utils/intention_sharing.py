import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gym.spaces import Discrete

from onpolicy.algorithms.utils.act import ACTLayer


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_hidden=2):
        nn.Module.__init__(self)
        activation_fn = nn.ReLU
        layers = [nn.Linear(input_size, hidden_size), activation_fn()]
        for _ in range(num_hidden):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation_fn()])
        layers.append(nn.Linear(hidden_size, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentionModule(nn.Module):
    def __init__(self, embed_dim, qdim=None, kdim=None, vdim=None):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_weights = nn.Linear(self.qdim, embed_dim)
        self.k_weights = nn.Linear(self.kdim, embed_dim)
        self.v_weights = nn.Linear(self.vdim, embed_dim)

    def forward(self, input_q, input_k, input_v):
        q = self.q_weights(input_q)
        k = self.k_weights(input_k)
        v = self.v_weights(input_v)
        # Add the target sequence length dimension
        q = q.unsqueeze(-2)
        # Get rid of the target sequence length dimension again
        return F.scaled_dot_product_attention(q, k, v).squeeze(-2)


class IntentionSharingModel(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        group_size,
        message_size=32,
        intention_aggregation='attention',
    ):
        nn.Module.__init__(self)

        self.obs_size = int(np.product(obs_space.shape))
        if isinstance(action_space, Discrete):
            self.action_size = action_space.n
        else:
            self.action_size = int(np.product(action_space.shape))
        self.hidden_size = 128
        self.num_hidden = 2
        self.group_size = group_size
        self.message_size = message_size
        self.intention_aggregation = intention_aggregation
        self.imagined_traj_len = 5

        # Takes an observation and other agent's messages. Returns a probability distribution over actions.
        self.policy = MLP(
            self.obs_size + (self.group_size - 1) * self.message_size,
            self.action_size,
            self.hidden_size,
            self.num_hidden,
        )

        # Takes an observation and returns a probability distribution over actions.
        self.action_predictor = MLP(
            self.obs_size,
            (self.group_size - 1) * self.action_size,
            self.hidden_size,
            self.num_hidden,
        )

        # Takes the last messages of other agents, the current agent's observation and action and the output of the action predictor. It predicts the agent's next observation.
        self.observation_predictor = MLP(
            self.group_size * self.action_size + self.obs_size,
            self.obs_size,
            self.hidden_size,
            self.num_hidden,
        )

        self.attention_module = AttentionModule(
            embed_dim=self.message_size,
            qdim=(self.group_size - 1) * self.message_size,
            kdim=self.obs_size + self.action_size,
            vdim=self.obs_size + self.action_size,
        )

    def forward(self, last_obs, other_messages=None):
        # If no messages are given, create starting value
        if other_messages is None:
            other_messages = torch.zeros(
                (*last_obs.shape[:-1], (self.group_size - 1) * self.message_size)
            )
        last_actions = self.policy(torch.cat((last_obs, other_messages), dim=-1))
        message = self._message_generation_network(
            last_obs, last_actions, other_messages
        )
        return last_actions, message

    def _predict_other_actions(self, obs):
        other_actions_raw = self.action_predictor(obs)
        other_actions = F.softmax(other_actions_raw.view(-1, self.action_size), dim=1)
        return other_actions.view(other_actions_raw.shape)

    def _imagined_trajectory_generation_module(
        self, last_obs, last_actions, other_messages
    ):
        predicted_last_other_actions = self._predict_other_actions(last_obs)
        predicted_obs = (
            self.observation_predictor(
                torch.cat(
                    (last_obs, last_actions, predicted_last_other_actions), dim=-1
                )
            )
            # The observation predictor soes only predict the difference between the next and current observations, hence we add the last ones to it.
            + last_obs
        )
        predicted_actions = self.policy(
            torch.cat((predicted_obs, other_messages), dim=-1)
        )
        return predicted_obs, predicted_actions

    def _message_generation_network(self, last_obs, last_actions, other_messages):
        trajectory = [torch.cat((last_obs, last_actions), dim=-1)]

        for _ in range(self.imagined_traj_len - 1):
            last_obs, last_actions = self._imagined_trajectory_generation_module(
                last_obs, last_actions, other_messages
            )
            trajectory.append(torch.cat((last_obs, last_actions), dim=-1))
        trajectory = torch.stack(trajectory)
        # We have to transpose the tensor to put the batch dim as the first and traj dim as second last
        dims = torch.arange(trajectory.dim())
        trajectory = trajectory.permute((*dims[1:-1], 0, -1))

        message = self.attention_module(other_messages, trajectory, trajectory)
        return message


class CTDEMultiAgentWrapper(nn.Module):
    def __init__(self, agent_model, agent_dim=-2):
        super().__init__()
        self.agent_model = agent_model
        self.message_size = agent_model.message_size
        self.agent_dim = agent_dim

    def forward(self, observations, messages=None):
        dim_size = observations.dim()
        agent_dim = dim_size + self.agent_dim if self.agent_dim < 0 else self.agent_dim
        num_agents = observations.shape[agent_dim]

        # If no messages are given, create starting value
        if messages is None:
            messages = torch.zeros((*observations.shape[:-1], self.message_size))

        actions = []
        next_messages = []
        # Break the input apart and feed it separately into the single agent model
        for idx in range(num_agents):
            # Create a mask that only returns the agent's observation
            agent_mask = tuple(
                slice(idx, idx + 1) if k == agent_dim else slice(None)
                for k in range(dim_size)
            )
            # Since the model needs the other agents' messages it needs a negative agent mask for the messages.
            message_mask = torch.ones_like(messages)
            message_mask[agent_mask] = 0
            message_mask = message_mask == 1
            # The truth mask removes dimansional information, hence we need to define the shape
            other_messages_dim = []
            for k in range(dim_size - 1):
                if k != agent_dim:
                    other_messages_dim.append(messages.shape[k])
            other_messages_dim.append(-1)
            other_messages_dim = tuple(other_messages_dim)
            # Extract the single agent's information from the input
            obs = observations[agent_mask].squeeze(agent_dim)
            other_messages = messages[message_mask].reshape(other_messages_dim)
            # Pass it through the single agent model
            action, next_message = self.agent_model(obs, other_messages)
            actions.append(action)
            next_messages.append(next_message)
        actions = torch.cat(actions, dim=-1)
        actions = actions.reshape((*observations.shape[:-1], -1))
        next_messages = torch.cat(next_messages, dim=-1)
        next_messages = next_messages.reshape((*observations.shape[:-1], -1))

        return actions, next_messages
