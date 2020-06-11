from utils.distributions import (CategoricalHead, MixtureGaussianHead,
                                 DiagGaussianHead, ScalarHead, MixedDist)
from agents.a2c.network import ActorCritic, MLP


class GMAC(ActorCritic):
    def __init__(self, input_shape, n_actions, min_var=0.0, **kwargs):
        super().__init__(input_shape, n_actions, min_var=min_var, **kwargs)
        if len(input_shape) == 1:
            input_shape = input_shape[0]
            self.enc_size = 512
            self.enc = MLP(input_shape, self.enc_size,
                           hidden_dim=[256, 256], act=True)

        if self.disc:
            self.action_head = CategoricalHead(self.enc_size, n_actions)
        else:
            action_head = DiagGaussianHead(self.enc_size, n_actions,
                                           min_var=min_var,
                                           lim=self.ac_space)
            self.action_head = action_head

        value_head = MixtureGaussianHead(self.enc_size, 1, n_mix=5,
                                         min_var=min_var)
        self.value_head = value_head
        self.intrinsic = hasattr(self, 'ivalue_head')

        self._init_params()
        self._init_params(module=self.action_head, val=0.1)

    def forward(self, x):
        z = self.enc(x)
        ac_dist = self.action_head(z)
        if self.sep and not self.image_input:
            z = self.enc_crit(x)
        val_dist = self.value_head(z)
        if self.intrinsic:
            ival_dist = self.ivalue_head(z)
            val_dist = MixedDist(val_dist, [ival_dist])
        return ac_dist, val_dist
