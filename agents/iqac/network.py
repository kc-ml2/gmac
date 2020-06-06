import numpy as np
import torch
import torch.nn as nn
from modules.deepmind import DeepMindEnc
from utils.distributions import CategoricalHead, DiagGaussianHead, SampleHead
from agents.a2c.network import ActorCritic, MLP


class IQPGActorCritic(ActorCritic):
    def __init__(self, input_shape, n_actions, n_quantiles,
                 hidden_dim=256, quantile_dim=64, disc=True, sep=False,
                 min_var=0.0, **kwargs):
        super().__init__(input_shape, n_actions, min_var=min_var, **kwargs)
        if len(input_shape) == 1:
            input_shape = input_shape[0]
            self.enc_size = 512
            self.enc = MLP(input_shape, self.enc_size,
                           hidden_dim=[256, 256], act=True)
            if sep:
                self.enc_crit = MLP(input_shape, self.enc_size,
                                    hidden_dim=[256, 256], act=True)

        self.n_quantiles = n_quantiles
        self.hidden_dim = hidden_dim
        self.quantile_dim = quantile_dim

        self.quantile_fc0 = nn.Linear(quantile_dim, self.enc_size)
        self.quantile_fc1 = nn.Linear(self.enc_size, hidden_dim)

        self.value_head = SampleHead(hidden_dim, 1)

        if disc:
            self.action_head = CategoricalHead(self.enc_size, n_actions)
        else:
            self.action_head = DiagGaussianHead(self.enc_size, n_actions,
                                                min_var=min_var,
                                                lim=self.ac_space)

        self._init_params()
        self._init_params(module=self.action_head, val=0.1)

    def encode_quantiles(self, z, taus):
        # create embedding for observation
        psi = z.unsqueeze(1).expand(-1, self.n_quantiles, -1)

        # create embedding for tau (cosine embedding)
        taus = taus.unsqueeze(-1)
        taus = taus.expand(-1, -1, self.quantile_dim)
        taus = taus.reshape(-1, self.quantile_dim).unsqueeze(1)
        qs = torch.FloatTensor(np.arange(1, self.quantile_dim + 1)).to(z)
        qs = qs.view(1, 1, -1)
        qs = qs.expand(z.size(0) * self.n_quantiles, 1, self.quantile_dim)

        tmp = torch.cos(qs * np.pi * taus).squeeze(1)
        phi = torch.relu(self.quantile_fc0(tmp))
        phi = phi.view(z.size(0), self.n_quantiles, -1)

        # combine two embeddings
        out = (psi * phi).view(z.size(0) * self.n_quantiles, -1)
        out = torch.relu(self.quantile_fc1(out))
        out = out.view(z.size(0), self.n_quantiles, -1)
        return out

    def forward(self, x, quantiles=None):
        z = self.enc(x)
        ac_dist = self.action_head(z)
        if quantiles is not None:
            taus = quantiles
        else:
            taus = torch.rand(z.size(0), self.n_quantiles).to(x)
        q = self.encode_quantiles(z, taus)
        val_dist = self.value_head(q)
        return ac_dist, val_dist
