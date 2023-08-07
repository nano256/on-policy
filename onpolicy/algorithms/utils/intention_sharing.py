import math
import torch
from torch import nn


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
