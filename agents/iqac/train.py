import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import settings
from agents.a2c.train import A2CAgent
from agents.iqpg.network import IQACActorCritic
from utils.common import load_model, save_model, safemean
from utils.summary import EvaluationMetrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
sns.set(style="white", palette="muted", color_codes=True)


__all__ = ['IQACAgent', 'IQACEAgent']


class IQACAgent(A2CAgent):
    def __init__(self, args, name='IQAC'):
        super().__init__(args, name)
        # Define constants
        self.vf_coef = self.args.vf_coef
        self.ent_coef = self.args.ent_coef
        self.cliprange = self.args.cliprange
        self.max_grad = 0.5
        self.n_quantiles = 64

        self.min_sig = 5e-2

        # Create policy
        if args.checkpoint is not None:
            path = os.path.join(settings.PROJECT_ROOT, settings.LOAD_DIR)
            path = os.path.join(path, args.checkpoint)
            model = load_model(path)
        else:
            input_shape = self.env.observation_space.shape
            if len(input_shape) > 1:
                input_shape = (input_shape[-1], *input_shape[:-1])
            if self.disc:
                n_actions = self.env.action_space.n
            else:
                n_actions = len(self.env.action_space.sample())
            ac_space = None
            model = IQACActorCritic(input_shape, n_actions,
                                    self.n_quantiles, disc=self.disc,
                                    ac=ac_space, min_var=self.min_sig**2)

        model = model.to(args.device)
        self.policy = model

        self.keys.append('vdists')
        self.buffer['vdists'] = []

        # Define optimizer
        self.optim = torch.optim.Adam(
            list(self.policy.parameters()),
            lr=args.lr
        )

        # Create statistics
        self.info = EvaluationMetrics(
            [
                'Time/Step',
                'Time/Item',
                'Loss/Total',
                'Loss/Value',
                'Loss/Policy',
                'Values/Entropy',
                'Values/Reward',
                'Values/Value',
                'Values/Adv',
                'Score/Train',
            ]
        )

        # Create histograms and graphs
        self.graph_dict = None
        self.huber = nn.SmoothL1Loss(reduction='none')

    def huber_quantile_loss(self, input, target, quantiles):
        target = target.detach()
        n_quantiles = quantiles.size(-1)
        diff = target.unsqueeze(2) - input.unsqueeze(1)
        taus = quantiles.unsqueeze(-1).unsqueeze(1)
        taus = taus.expand(-1, n_quantiles, -1, -1)
        huber = self.huber(input.unsqueeze(1), target.unsqueeze(2))
        loss = huber * (taus - (diff < 0).float()).abs()
        return loss.squeeze(3).sum(-1).mean(-1)

    def compute_returns(self, **kwargs):
        # Generalized Advantage Estimate
        gae = 0
        advs = torch.zeros(*self.buffer['vals'].shape[:-1],
                           1).to(self.args.device)
        dones = torch.from_numpy(self.dones).to(self.args.device)

        # Look one step further
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            _, val_dist = self.policy(obs, **kwargs)
        vals = val_dist.mean
        next_vals = vals

        rets = torch.zeros_like(self.buffer['vals'])
        for t in reversed(range(self.update_step)):
            if t == self.update_step - 1:
                _nont = 1.0 - dones.float()
                _vals = vals.mean(-1, keepdim=True)
            else:
                _nont = 1.0 - self.buffer['dones'][t + 1].float()
                _vals = self.buffer['vals'][t + 1].mean(-1, keepdim=True)

            while len(_nont.shape) < len(vals.shape):
                _nont = _nont.unsqueeze(1)
            rews = self.buffer['rews'][t]
            while len(rews.shape) < len(vals.shape):
                rews = rews.unsqueeze(1)

            # Bellman operator on samples
            next_vals *= _nont * self.gam
            next_vals += rews

            rets[t] = next_vals.clone()

            curr_vals = self.buffer['vals'][t]

            # Replace samples
            probs = self.lam * torch.ones_like(self.buffer['nlps'][t])
            dist = Bernoulli(probs=probs)
            mask = dist.sample((self.n_quantiles,)).to(vals)
            mask = mask.transpose(0, 1).squeeze(-1)
            next_vals += (1 - mask) * (curr_vals - next_vals)

            vals = self.buffer['vals'][t].mean(-1, keepdim=True)
            delta = rews + _nont * self.gam * _vals - vals
            gae = delta + _nont * self.gam * self.lam * gae
            advs[t] = gae

        return rets.detach(), advs.detach()

    def compute_loss(self, idx):
        # Compute action distributions
        obs = self.buffer['obs'][idx]
        acs = self.buffer['acs'][idx]
        # taus = self.buffer['taus'][idx]
        taus = torch.rand(obs.size(0), self.n_quantiles)
        taus = taus.to(self.args.device)
        ac_dist, val_dist = self.policy(obs, taus)
        vals = val_dist.mean.unsqueeze(-1)
        nlps = -ac_dist.log_prob(acs)
        ent = ac_dist.entropy().mean()
        self.info.update('Values/Value', vals.mean().item())
        self.info.update('Values/Entropy', ent.item())

        rets = self.buffer['rets'][idx].unsqueeze(-1)
        advs = self.buffer['advs'][idx]
        advs = (advs - advs.mean()) / (advs.std() + settings.EPS)
        self.info.update('Values/Adv', advs.max().item())

        vf_loss = self.huber_quantile_loss(vals, rets, taus)
        vf_loss = vf_loss.mean()
        self.info.update('Loss/Value', vf_loss.item())

        # Policy gradient with clipped ratio
        ratio = torch.exp(self.buffer['nlps'][idx] - nlps).unsqueeze(-1)
        pg_loss1 = -advs * ratio
        ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)
        pg_loss2 = -advs * ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        self.info.update('Loss/Policy', pg_loss.item())

        # Total loss
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent
        self.info.update('Loss/Total', loss.item())
        self.graph_dict = {
            'vals': vals[:20].squeeze().detach().cpu().numpy(),
            'rets': rets[:20].squeeze().detach().cpu().numpy(),
        }

        return loss

    def train(self):
        self.policy.train()
        st = time.time()
        self.step += 1

        _, val_dist = self.collect()
        self.buffer['vdists'].append(val_dist)
        if self.step % self.update_step == 0:
            for k, v in self.buffer.items():
                if k is not 'vdists' and k is not 'acdists':
                    v = torch.from_numpy(np.asarray(v))
                    self.buffer[k] = v.float().to(self.args.device)

            self.buffer['rets'], self.buffer['advs'] = self.compute_returns()
            for k, v in self.buffer.items():
                if k is not 'vdists' and k is not 'acdists':
                    self.buffer[k] = v.view(-1, *v.shape[2:])

            for _ in range(self.args.epoch):
                # Shuffle collected batch
                idx = np.arange(len(self.buffer['rets']))
                np.random.shuffle(idx)

                # Train from collected batch
                start = 0
                batch_size = self.args.batch_size
                for _ in range(len(self.buffer['rets']) // batch_size):
                    end = start + batch_size
                    _idx = np.array(idx[start:end])
                    start = end
                    loss = self.compute_loss(_idx)

                    self.optim.zero_grad()
                    loss.backward()
                    if self.max_grad is not None:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            self.max_grad
                        )
                    self.optim.step()

            # Clear buffer
            self.buffer = {k: [] for k in self.keys}

        elapsed = time.time() - st
        self.info.update('Time/Step', elapsed)
        self.info.update('Time/Item', elapsed/self.args.batch_size)
        if self.step % self.log_step == 0:
            self.info.update(
                'Score/Train',
                safemean([score for score in self.epinfobuf])
            )
            frames = self.step * self.args.num_workers
            self.logger.log(
                "[{}]\nTraining statistics for step: {}".format(self.args.tag,
                                                                frames)
            )
            self.logger.scalar_summary(
                self.info.avg,
                frames,
                tag='train'
            )
            # self.plot_graphs()
            self.info.reset()

        # Save model
        if (self.save_step is not None and
                self.step % self.save_step == 0):
            path = os.path.join(self.logger.log_dir, 'checkpoints')
            if not os.path.exists(path):
                os.makedirs(path)
            filename = str(int(self.step // self.save_step))
            path = os.path.join(path, filename + '.pth')
            save_model(self.policy, path)

    def plot_graphs(self):
        if self.graph_dict is not None:
            fig1 = plt.figure(1)
            fig2 = plt.figure(2)
            for gi in range(20):
                vals = self.graph_dict['vals'][gi]
                rets = self.graph_dict['rets'][gi]

                plt.figure(1)
                sns.kdeplot(vals)
                plt.figure(2)
                sns.kdeplot(rets)

            self.logger.image_summary(
                fig1,
                self.step * self.args.num_workers,
                tag='PDF/Value'
            )
            self.logger.image_summary(
                fig2,
                self.step * self.args.num_workers,
                tag='PDF/Target'
            )


class IQACEAgent(IQACAgent):
    def __init__(self, args, name='IQACE'):
        super().__init__(args, name)

    def compute_loss(self, idx):
        # Compute action distributions
        obs = self.buffer['obs'][idx]
        acs = self.buffer['acs'][idx]
        taus = torch.rand(obs.size(0), self.n_quantiles)
        taus = taus.to(self.args.device)
        ac_dist, val_dist = self.policy(obs, taus)
        vals = val_dist.mean
        nlps = -ac_dist.log_prob(acs)
        ent = ac_dist.entropy().mean()
        self.info.update('Values/Value', vals.mean().item())
        self.info.update('Values/Entropy', ent.item())

        advs = self.buffer['advs'][idx]
        advs = (advs - advs.mean()) / (advs.std() + settings.EPS)
        self.info.update('Values/Adv', advs.max().item())

        # Energy distance
        rets = self.buffer['rets'][idx]
        ridx = torch.randperm(vals.size(1)).to(vals).long()
        vf_dist = 2 * (vals - rets).abs().mean(-1)
        vf_dist -= (vals - vals[:, ridx]).abs().mean(-1)
        vf_dist -= (rets - rets[:, ridx]).abs().mean(-1)
        vf_loss = vf_dist.mean()
        self.info.update('Values/Value', vals.mean().item())
        self.info.update('Loss/Value', vf_loss.item())

        # Policy gradient with clipped ratio
        ratio = torch.exp(self.buffer['nlps'][idx] - nlps).unsqueeze(-1)
        pg_loss1 = -advs * ratio
        ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)
        pg_loss2 = -advs * ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        self.info.update('Loss/Policy', pg_loss.item())

        # Total loss
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent
        self.info.update('Loss/Total', loss.item())
        self.graph_dict = {
            'vals': vals[:20].squeeze().detach().cpu().numpy(),
            'rets': rets[:20].squeeze().detach().cpu().numpy(),
        }

        return loss
