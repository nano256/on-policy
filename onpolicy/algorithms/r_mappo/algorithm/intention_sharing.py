import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gym.spaces import Discrete

from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import init, check


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_hidden=2):
        nn.Module.__init__(self)
        activation_fn = nn.ReLU
        layers = [
            nn.Linear(input_size, hidden_size),
            activation_fn(),
            nn.LayerNorm(hidden_size),
        ]
        for _ in range(num_hidden):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    activation_fn(),
                    nn.LayerNorm(hidden_size),
                ]
            )
        layers.append(nn.Linear(hidden_size, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentionModule(nn.Module):
    def __init__(
        self, embed_dim, qdim=None, kdim=None, vdim=None, value_transform="learned"
    ):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_weights = nn.Linear(self.qdim, embed_dim)
        self.k_weights = nn.Linear(self.kdim, embed_dim)
        if value_transform == "learned":
            self.v_weights = nn.Linear(self.vdim, embed_dim)
        elif value_transform == "identity":
            assert (
                embed_dim == self.vdim
            ), "embed and v dim must be identical for unity mod of value transform."
            self.v_weights = nn.Identity()
        else:
            ValueError(f'"{value_transform}" is not a valid value transformation mode.')

    def forward(self, input_q, input_k, input_v):
        q = self.q_weights(input_q)
        k = self.k_weights(input_k)
        v = self.v_weights(input_v)
        # Add the target sequence length dimension
        q = q.unsqueeze(-2)
        # Get rid of the target sequence length dimension again
        return self._scaled_dot_product_attention(q, k, v).squeeze(-2)

    def _scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ) -> torch.Tensor:
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(device=query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value


class IntentionSharingModel(nn.Module):
    def __init__(
        self,
        args,
        obs_space,
        action_space,
        message_space,
        n_agents,
        device=torch.device("cpu"),
    ):
        nn.Module.__init__(self)
        self.hidden_size = args.hidden_size
        self.num_hidden = 2
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.intention_aggregation = args.intention_aggregation
        self.imagined_traj_len = args.imagined_traj_len
        self.communication_interval = args.communication_interval
        self.group_size = n_agents

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.last_imagined_traj = None

        self.obs_size = int(np.product(obs_space.shape))
        if isinstance(action_space, Discrete):
            self.action_size = action_space.n
        else:
            self.action_size = int(np.product(action_space.shape))
        self.message_size = int(np.product(message_space.shape))

        self.base = MLPBase(
            args, (self.obs_size + (self.group_size - 1) * self.message_size,)
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_size,
            self._use_orthogonal,
            self._gain,
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

        if self.intention_aggregation == "attention":
            self.attention_module = AttentionModule(
                embed_dim=self.message_size,
                qdim=(self.group_size - 1) * self.message_size,
                kdim=self.obs_size + self.action_size,
                vdim=self.obs_size + self.action_size,
                value_transform=args.msg_value_transformation,
            )
        elif self.intention_aggregation == "mean":
            self.attention_module = lambda t: torch.mean(t, -2)
        else:
            ValueError(
                f'"{self.intention_aggregation}" is not a valid intention aggregation mode.'
            )
        self.to(device)

    def forward(
        self,
        obs,
        messages,
        rnn_states,
        masks,
        step,
        available_actions=None,
        deterministic=False,
    ):
        last_obs = check(obs).to(**self.tpdv)
        messages = check(messages).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.reshape(
                (-1, available_actions.size(-1))
            )
        last_obs = last_obs.reshape((-1, last_obs.size(-1)))
        masks = masks.reshape((-1, masks.size(-1)))
        rnn_states = rnn_states.reshape((-1, *rnn_states.shape[2:]))

        # We want the other agents' messages to be included in the agent's observations
        own_messages, other_messages = self._preprocess_messages(messages)
        own_messages = own_messages.reshape((-1, own_messages.size(-1)))
        other_messages = other_messages.reshape((-1, other_messages.size(-1)))

        last_actions, action_log_probs, rnn_states = self.policy(
            torch.cat((last_obs, other_messages), dim=-1),
            rnn_states,
            masks,
            deterministic,
        )
        if step % self.communication_interval == 0:
            message, trajectory = self._message_generation_network(
                last_obs, last_actions, other_messages, rnn_states, masks, deterministic
            )
        else:
            message = own_messages
            trajectory = None
        return last_actions, action_log_probs, message, rnn_states, trajectory

    def evaluate_actions(
        self,
        obs,
        messages,
        rnn_states,
        action,
        masks,
        steps,
        available_actions=None,
        active_masks=None,
        get_action_probs=False,
    ):
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
            # Steps gives back the timestep of the batch, not the actor. Hence repeat and reshape.
            steps = np.tile(steps.reshape(-1, 1), (1, self.group_size)).reshape(-1)

            action_log_probs_batch = []
            dist_entropy_batch = []
            action_probs_batch = []
            _, last_message = self._preprocess_messages(messages)
            last_message = last_message.reshape(-1, last_message.size(-1))
            # `rnn_states` contains the initial state of the RNN, no need for slicing
            last_rnn_state = rnn_states.reshape((-1, *rnn_states.shape[2:]))

        for ts in range(obs.shape[0]):
            input = torch.cat((obs[ts, ...], last_message), dim=-1)

            action_log_probs, dist_entropy = self.policy_evaluate_actions(
                input,
                last_rnn_state,
                action[ts, ...],
                available_actions[ts, ...],
                masks[ts, ...],
                active_masks=active_masks[ts, ...]
                if self._use_policy_active_masks
                else None,
            )
            action_log_probs_batch.append(action_log_probs)
            dist_entropy_batch.append(dist_entropy)

            action_probs = self.policy_get_action_probs(
                input, last_rnn_state, masks[ts, ...]
            )
            action_probs_batch.append(action_probs)

            last_actions, _, last_rnn_state = self.policy(
                input, last_rnn_state, masks[ts, ...]
            )
            new_message, _ = self._message_generation_network(
                obs[ts, ...], last_actions, last_message, last_rnn_state, masks[ts, ...]
            )
            # The messages have to be redistributed to the different agents
            _, new_message = self._preprocess_messages(
                new_message.reshape(messages.shape)
            )
            new_message = new_message.reshape((-1, new_message.size(-1)))
            # Get a mask that only gets the messages where a new communication interval starts
            msg_update_mask = (steps + ts) % self.communication_interval == 0
            # We have to merge old and new values in a new tensor without backward history, otherwise it will cause an error
            msg_placeholder = torch.zeros_like(last_message)
            msg_placeholder[msg_update_mask] = new_message[msg_update_mask]
            msg_placeholder[~msg_update_mask] = last_message[~msg_update_mask]
            last_message = msg_placeholder

        action_log_probs = torch.stack(action_log_probs_batch).reshape((*dims, 1))
        action_probs = torch.stack(action_probs_batch).reshape((*dims, -1))
        dist_entropy = torch.stack(dist_entropy_batch).mean()
        if get_action_probs:
            return action_log_probs, dist_entropy, action_probs
        else:
            return action_log_probs, dist_entropy

    def policy(self, input, rnn_states=None, masks=None, deterministic=False):
        x = self.base(input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)

        actions, action_log_probs = self.act(x, deterministic=deterministic)
        return actions, action_log_probs, rnn_states

    def policy_evaluate_actions(
        self, input, rnn_states, action, available_actions, masks, active_masks=None
    ):
        x = self.base(input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)

        return self.act.evaluate_actions(x, action, available_actions, active_masks)

    def policy_get_action_probs(self, input, rnn_states, masks):
        x = self.base(input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)
        return self.act.get_probs(x)

    def _predict_other_actions(self, obs):
        other_actions_raw = self.action_predictor(obs)
        other_actions = torch.argmax(
            other_actions_raw.reshape(-1, self.action_size), dim=1
        )
        other_actions = self._one_hot_actions(other_actions)
        return other_actions.reshape(other_actions_raw.shape)

    # We don't want the message generation to affect the world model params
    @torch.no_grad()
    def _imagined_trajectory_generation_module(
        self, last_obs, last_actions, other_messages, rnn_states, masks, deterministic
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
        predicted_actions, _, rnn_states = self.policy(
            torch.cat((predicted_obs, other_messages), dim=-1),
            rnn_states,
            masks,
            deterministic,
        )
        return predicted_obs, predicted_actions, rnn_states

    def _message_generation_network(
        self,
        last_obs,
        last_actions,
        other_messages,
        rnn_states,
        masks,
        deterministic=False,
    ):
        last_actions = self._one_hot_actions(last_actions)
        trajectory = [torch.cat((last_obs, last_actions), dim=-1)]

        for _ in range(self.imagined_traj_len - 1):
            (
                last_obs,
                last_actions,
                rnn_states,
            ) = self._imagined_trajectory_generation_module(
                last_obs,
                last_actions,
                other_messages,
                rnn_states,
                masks,
                deterministic=False,
            )
            last_actions = self._one_hot_actions(last_actions)
            trajectory.append(torch.cat((last_obs, last_actions), dim=-1))
        trajectory = torch.stack(trajectory).to(**self.tpdv)
        # We have to transpose the tensor to put the batch dim as the first and traj dim as second last
        dims = torch.arange(trajectory.dim())
        trajectory = trajectory.permute((*dims[1:-1], 0, -1))

        if self.intention_aggregation == "attention":
            message = self.attention_module(other_messages, trajectory, trajectory)
        elif self.intention_aggregation == "mean":
            message = self.attention_module(trajectory)
        return message, trajectory

    def _one_hot_actions(self, actions):
        return F.one_hot(actions, self.action_size).squeeze().float()

    def _preprocess_messages(self, messages):
        if messages.dim() == 3:
            # We assume messages having the dims (Batch, Agent, Features)
            msg_shape = messages.shape
            n_agents = messages.shape[1]
            own_messages = []
            other_messages = []
            for idx in range(n_agents):
                mask = torch.ones_like(messages)
                mask[:, idx, :] = 0
                mask = mask == 1
                own_messages.append(messages[~mask].reshape(msg_shape[0], 1, -1))
                other_messages.append(messages[mask].reshape(msg_shape[0], 1, -1))
            own_messages = torch.cat(own_messages, 1)
            other_messages = torch.cat(other_messages, 1)
        elif messages.dim() == 4:
            # We assume messages having the dims (Seq, Batch, Agent, Features)
            msg_shape = messages.shape
            n_agents = messages.shape[2]
            own_messages = []
            other_messages = []
            for idx in range(n_agents):
                mask = torch.ones_like(messages)
                mask[:, :, idx, :] = 0
                mask = mask == 1
                own_messages.append(messages[~mask].reshape(*msg_shape[:2], 1, -1))
                other_messages.append(messages[mask].reshape(*msg_shape[:2], 1, -1))
            own_messages = torch.cat(own_messages, 2)
            other_messages = torch.cat(other_messages, 2)
        else:
            ValueError(
                f"Input expected to hva 3 or 4 dimensions, got {messages.dim()} instead."
            )
        return own_messages, other_messages


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
