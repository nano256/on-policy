import torch.nn as nn
from onpolicy.algorithms.utils.intention_sharing import MLP, AttentionModule


class EncoderDecoderModule(nn.Module):
    def __init__(self, input_size, latent_space_size, args):
        super().__init__()
        self.encoder = MLP(
            input_size, latent_space_size, args.hidden_size, args.layer_N
        )
        self.decoder = MLP(
            latent_space_size, input_size, args.hidden_size, args.layer_N
        )

    def forward(self, x):
        z = self.enc_forward(x)
        return self.dec_forward(z)

    def enc_forward(self, x):
        return self.encoder(x)

    def dec_forward(self, z):
        return self.decoder(z)
