import torch
import settings
from agents.a2c.train import A2CAgent


__all__ = ['PPOAgent']


class PPOAgent(A2CAgent):
    def __init__(self, args, name='PPO'):
        super().__init__(args, name)
        # Define constants
        self.vf_coef = self.args.vf_coef
        self.ent_coef = self.args.ent_coef
        self.cliprange = self.args.cliprange
        self.max_grad = 0.5
        self.clip_value = True

        # Define optimizer
        self.optim = torch.optim.Adam(
            self.policy.parameters(),
            lr=args.lr,
            eps=1e-5
        )

    def compute_loss(self, idx):
        # Compute action distributions
        obs = self.buffer['obs'][idx]
        acs = self.buffer['acs'][idx]
        ac_dist, val_dist = self.policy(obs)
        vals = val_dist.mean
        nlps = -ac_dist.log_prob(acs)
        ent = ac_dist.entropy().mean()
        self.info.update('Values/Value', vals.mean().item())
        self.info.update('Values/Entropy', ent.item())

        advs = self.buffer['advs'][idx].detach()
        rets = self.buffer['vals'][idx] + advs
        advs = (advs - advs.mean()) / (advs.std() + settings.EPS)
        self.info.update('Values/Adv', advs.max().item())

        if self.clip_value:
            old_vals = self.buffer['vals'][idx]
            vals_clipped = (old_vals + torch.clamp(vals - old_vals,
                                                   -self.cliprange,
                                                   self.cliprange))
            vf_loss_clipped = 0.5 * (vals_clipped - rets.detach()).pow(2)
            vf_loss = 0.5 * (vals - rets.detach()).pow(2)
            vf_loss = torch.max(vf_loss, vf_loss_clipped).mean()
        else:
            vf_loss = 0.5 * (rets.detach() - vals).pow(2).mean()
        self.info.update('Loss/Value', vf_loss.item())

        # Policy gradient with clipped ratio
        ratio = torch.exp(self.buffer['nlps'][idx] - nlps).unsqueeze(-1)
        pg_loss1 = -advs * ratio
        ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)
        pg_loss2 = -advs * ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        self.info.update('Loss/Policy', pg_loss.item())

        # Total loss
        loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss
        self.info.update('Loss/Total', loss.item())
        return loss
