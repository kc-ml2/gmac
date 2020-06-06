import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from modules.deepmind import DeepMindEnc
from utils.distributions import CategoricalHead, DiagGaussianHead, ScalarHead


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions, disc=True, sep=False,
                 ac=None, min_var=0.0, **kwargs):
        super().__init__()
        self.n_actions = n_actions

        # Define encoder accordingly
        # disc indicates discrete action space
        self.disc = disc
        self.sep = sep
        self.ac_space = ac
        if len(input_shape) == 3:
            self.image_input = True
            self.enc = DeepMindEnc(input_shape)
            sample = torch.FloatTensor(1, *input_shape)
            self.enc_size = self.enc(sample).size(-1)
        elif len(input_shape) == 1:
            self.image_input = False
            input_shape = input_shape[0]
            self.enc_size = 512
            self.enc = MLP(input_shape, self.enc_size,
                           hidden_dim=[256, 256], act=True)
            if self.sep:
                self.enc_crit = MLP(input_shape, self.enc_size,
                                    hidden_dim=[256, 256], act=True)
        else:
            raise ValueError("Unhandled input shape")

        if disc:
            self.action_head = CategoricalHead(self.enc_size, n_actions)
        else:
            self.action_head = DiagGaussianHead(self.enc_size, n_actions,
                                                min_var=min_var,
                                                lim=self.ac_space)

        self.value_head = ScalarHead(self.enc_size, 1)

        self._init_params()
        self._init_params(module=self.action_head, val=0.1)

    def _init_params(self, module=None, val=np.sqrt(2)):
        # Weight initialization
        if module is not None:
            target = module.modules()
        else:
            target = self.modules()
        for p in target:
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, val)
                p.bias.data.zero_()
            elif isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, val)
                p.bias.data.zero_()

    def forward(self, x):
        z = self.enc(x)
        ac_dist = self.action_head(z)
        if self.sep and not self.image_input:
            z = self.enc_crit(x)
        val_dist = self.value_head(z)
        return ac_dist, val_dist


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim=[128], act=False):
        super().__init__()
        mod_list = nn.ModuleList([])
        prev_shape = input_shape
        for dim in hidden_dim:
            mod_list.append(nn.Linear(prev_shape, dim))
            # mod_list.append(nn.ReLU())
            mod_list.append(nn.Tanh())
            prev_shape = dim
        mod_list.append(nn.Linear(prev_shape, output_shape))
        if act:
            # mod_list.append(nn.ReLU())
            mod_list.append(nn.Tanh())

        self.body = nn.Sequential(*mod_list)

    def forward(self, x):
        return self.body(x)
