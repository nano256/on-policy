import numpy as np
import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class CR_Actor(nn.Module):
    """
    Actor network class for MAPPO extended with inter-agent communication. Outputs actions and messages, given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        message_space,
        n_agents,
        device=torch.device("cpu"),
    ):
        super(CR_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._message_N = 2
        self.n_agents = n_agents
        self._message_size = int(np.product(message_space.shape))
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        msg_shape = get_shape_from_obs_space(message_space)

        # This model can only work with 1D inputs
        assert (
            len(obs_shape) == 1
        ), f"Observation shape must be 1D, but has {len(obs_shape)} dimensions."
        assert (
            len(msg_shape) == 1
        ), f"Message shape must be 1D, but has {len(msg_shape)} dimensions."
        self.base = MLPBase(args, (obs_shape[0] + msg_shape[0] * (self.n_agents - 1),))

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        msg_layers = []
        for _ in range(self._message_N):
            msg_layers.extend(
                [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()]
            )
        msg_layers.append(nn.Linear(self.hidden_size, self._message_size))
        self.msg = nn.Sequential(*msg_layers)

        self.to(device)

    def forward(
        self,
        obs,
        messages,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        messages = check(messages).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.reshape(
                (-1, available_actions.size(-1))
            )
        masks = masks.reshape((-1, masks.size(-1)))
        rnn_states = rnn_states.reshape((-1, *rnn_states.shape[2:]))

        # We want the other agents' messages to be included in the agent's observations
        other_messages = self.preprocess_messages(messages)
        input = torch.cat((obs, other_messages), -1)
        input = input.reshape((-1, input.size(-1)))

        actor_features = self.base(input)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        messages = self.msg(actor_features)

        return actions, action_log_probs, messages, rnn_states

    def evaluate_actions(
        self,
        obs,
        messages,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param messages: (torch.Tensor) messages of other agents.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        messages = check(messages).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # assert obs.dim() == 4, f'We assume input tensors having the dims (Seq, Batch, Agent, Features), only had {messages.dim()}.'

        if obs.size(0) == rnn_states.size(0):
            ValueError("I guess this shouldn't happen...")
        else:
            dims = masks.shape[:-1]
            # Get rid of the actor dimension where we don't need it anymore
            action = action.reshape((action.size(0), -1, action.size(-1)))
            masks = masks.reshape((masks.size(0), -1, masks.size(-1)))
            obs = obs.reshape((obs.size(0), -1, obs.size(-1)))
            if available_actions is not None:
                available_actions = available_actions.reshape(
                    (available_actions.size(0), -1, available_actions.size(-1))
                )
            if active_masks is not None:
                active_masks = active_masks.reshape(
                    (active_masks.size(0), -1, active_masks.size(-1))
                )

            action_log_probs_batch = []
            dist_entropy_batch = []
            last_message = self.preprocess_messages(messages)
            last_message = last_message.reshape(-1, last_message.size(-1))
            # `rnn_states` contains the initial state of the RNN, no need for slicing
            last_rnn_state = rnn_states.reshape((-1, *rnn_states.shape[2:]))

        for ts in range(obs.shape[0]):
            actor_features = self.base(torch.cat((obs[ts, ...], last_message), -1))

            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, last_rnn_state = self.rnn(
                    actor_features, last_rnn_state, masks[ts, ...]
                )

            # The messages have to be redistributed to the different agents
            last_message = self.msg(actor_features).reshape(messages.shape)
            last_message = self.preprocess_messages(last_message)
            last_message = last_message.reshape((-1, last_message.size(-1)))

            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action[ts, ...],
                available_actions[ts, ...],
                active_masks=active_masks[ts, ...]
                if self._use_policy_active_masks
                else None,
            )
            action_log_probs_batch.append(action_log_probs)
            dist_entropy_batch.append(dist_entropy)

        action_log_probs = torch.stack(action_log_probs_batch).reshape((*dims, 1))
        dist_entropy = torch.stack(dist_entropy_batch).mean()

        return action_log_probs, dist_entropy

    def preprocess_messages(self, messages):
        if messages.dim() == 3:
            # We assume messages having the dims (Batch, Agent, Features)
            msg_shape = messages.shape
            n_agents = messages.shape[1]
            other_messages = []
            for idx in range(n_agents):
                mask = torch.ones_like(messages)
                mask[:, idx, :] = 0
                mask = mask == 1
                other_messages.append(messages[mask].reshape(msg_shape[0], 1, -1))
            other_messages = torch.cat(other_messages, 1)
        elif messages.dim() == 4:
            # We assume messages having the dims (Seq, Batch, Agent, Features)
            msg_shape = messages.shape
            n_agents = messages.shape[2]
            other_messages = []
            for idx in range(n_agents):
                mask = torch.ones_like(messages)
                mask[:, :, idx, :] = 0
                mask = mask == 1
                other_messages.append(messages[mask].reshape(*msg_shape[:2], 1, -1))
            other_messages = torch.cat(other_messages, 2)
        else:
            ValueError(
                f"Input expected to hva 3 or 4 dimensions, got {messages.dim()} instead."
            )
        return other_messages


class CR_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(CR_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        dims = masks.shape[:-1]
        rnn_dims = rnn_states.shape
        cent_obs = cent_obs.reshape((-1, cent_obs.size(-1)))
        rnn_states = rnn_states.reshape((-1, *rnn_states.shape[2:]))
        masks = masks.reshape((-1, masks.size(-1)))

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
