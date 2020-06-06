import os
import time
import numpy as np
import torch
import torch.nn as nn
import settings
from utils.common import save_model, load_model, safemean
from agents.a2c.train import A2CAgent
from agents.gmac.network import GMAC
from torch.distributions import Bernoulli, Normal, Categorical
from utils.loss import Cramer
from utils.summary import EvaluationMetrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
sns.set(style="white", palette="muted", color_codes=True)


__all__ = ['GMACAgent']


class GMACAgent(A2CAgent):
    def __init__(self, args, name='GMAC'):
        super().__init__(args, name)
        # Define constants
        self.vf_coef = self.args.vf_coef
        self.ent_coef = self.args.ent_coef
        self.cliprange = self.args.cliprange
        # self.max_grad = None
        self.max_grad = 5.0

        self.n_sample = 128
        self.min_sig = 5e-2

        # Create policy and buffer
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
                ac_space = None
            else:
                n_actions = len(self.env.action_space.sample())
                ac_space = (self.env.action_space.low,
                            self.env.action_space.high)
                ac_space = None
            model = GMAC(input_shape, n_actions, disc=self.disc, ac=ac_space,
                         min_var=self.min_sig**2)

        model = model.to(args.device)
        self.policy = model

        # Add value distribution to the buffer
        self.keys.append('vdists')
        self.keys.append('acdists')
        self.buffer['vdists'] = []
        self.buffer['acdists'] = []

        # Define optimizer
        # self.optim = torch.optim.RMSprop(
        self.optim = torch.optim.Adam(
            self.policy.parameters(),
            lr=args.lr,
            eps=1e-5
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
                'Values/VEntropy',
                'Values/WEntropy',
                'Values/Reward',
                'Values/Value',
                'Values/Adv',
                'Score/Train',
            ]
        )

    def compute_returns(self, **kwargs):
        gae = 0
        advs = torch.zeros_like(self.buffer['vals'])
        dones = torch.from_numpy(self.dones).to(self.args.device)
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            _, val_dist = self.policy(obs, **kwargs)
        vals = val_dist.mean
        next_mus, next_sigs = val_dist.sample_param(self.n_sample)

        size = (self.update_step, self.args.num_workers, self.n_sample)
        rets_mus = torch.zeros(*size).to(vals)
        rets_sigs = torch.zeros(*size).to(vals)
        for t in reversed(range(self.update_step)):
            if t == self.update_step - 1:
                _nont = 1.0 - dones.float()
                _vals = vals
            else:
                _nont = 1.0 - self.buffer['dones'][t + 1].float()
                _vals = self.buffer['vals'][t + 1]

            while len(_nont.shape) < len(vals.shape):
                _nont = _nont.unsqueeze(1)
            rews = self.buffer['rews'][t]
            while len(rews.shape) < len(vals.shape):
                rews = rews.unsqueeze(1)

            next_mus *= _nont * self.gam
            next_mus += rews
            next_sigs *= _nont * self.gam

            rets_mus[t] = next_mus.clone()
            rets_sigs[t] = next_sigs.clone()

            curr_val_dist = self.buffer['vdists'][t]
            curr_mus, curr_sigs = curr_val_dist.sample_param(self.n_sample)

            # Probability of next state
            pi = torch.exp(-self.buffer['nlps'][t])
            probs = self.lam * torch.ones_like(pi)
            dist = Bernoulli(probs=probs)
            mask = dist.sample((self.n_sample,)).to(vals)
            mask = mask.transpose(0, 1).squeeze(-1)
            next_mus += (1 - mask) * (curr_mus - next_mus)
            next_sigs += (1 - mask) * (curr_sigs - next_sigs)

            vals = self.buffer['vals'][t]
            delta = rews + _nont * self.gam * _vals - vals
            gae = delta + _nont * self.gam * self.lam * gae
            advs[t] = gae

        return rets_mus.detach(), rets_sigs.detach(), advs.detach()

    def compute_loss(self, idx):
        # Compute action distributions
        obs = self.buffer['obs'][idx]
        acs = self.buffer['acs'][idx]
        ac_dist, val_dist = self.policy(obs)
        vals = val_dist.sample(self.n_sample)
        nlps = -ac_dist.log_prob(acs)
        ent = ac_dist.entropy().mean()
        vent = val_dist.entropy().mean()
        went = Categorical(probs=val_dist.w).entropy().mean()
        self.info.update('Values/Value', vals.mean().item())
        self.info.update('Values/Entropy', ent.item())
        self.info.update('Values/VEntropy', vent.item())
        self.info.update('Values/WEntropy', went.item())

        _rets = self.buffer['rets'][idx]
        advs = self.buffer['advs'][idx]
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)
        self.info.update('Values/Adv', advs.max().item())

        vf_loss = Cramer(val_dist.loc,
                         val_dist.scale,
                         val_dist.w,
                         self.buffer['ret_mus'][idx],
                         self.buffer['ret_sigs'][idx],
                         1 / self.n_sample * torch.ones_like(_rets)).mean()
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
        self.graph_dict = {
            'vals': vals[:20].detach().cpu().numpy(),
            'rets': _rets[:20].detach().cpu().numpy(),
        }

        return loss

    def train(self, **kwargs):
        self.policy.train()
        st = time.time()
        self.step += 1

        ac_dist, val_dist = self.collect(**kwargs)
        if self.intrinsic:
            irew = torch.exp(val_dist.entropy())
            self.i_rms.update(irew.detach(), torch.from_numpy(
                self.buffer['dones'][-1]).to(irew))
        self.buffer['vdists'].append(val_dist)
        self.buffer['acdists'].append(ac_dist)
        if self.step % self.update_step == 0:
            for k, v in self.buffer.items():
                if k is not 'vdists' and k is not 'acdists':
                    v = torch.from_numpy(np.asarray(v))
                    self.buffer[k] = v.float().to(self.args.device)
            self.buffer['ret_mus'], self.buffer['ret_sigs'], \
                self.buffer['advs'] = self.compute_returns(**kwargs)
            self.buffer['rets'] \
                = Normal(self.buffer['ret_mus'],
                         self.buffer['ret_sigs']).sample()
            for k, v in self.buffer.items():
                if k is not 'vdists' and k is not 'acdists':
                    self.buffer[k] = v.view(-1, *v.shape[2:])

            for _ in range(self.args.epoch):
                # Shuffle collected batch
                idx = np.arange(len(self.buffer['ret_mus']))
                np.random.shuffle(idx)

                # Train from collected batch
                start = 0
                for _ in range(len(self.buffer['ret_mus'])
                               // self.args.batch_size):
                    end = start + self.args.batch_size
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
        self.info.update('Time/Item', elapsed / self.args.num_workers)
        if self.step % self.log_step == 0:
            self.video_flag = True and self.args.playback
            self.info.update(
                'Score/Train',
                safemean([score for score in self.epinfobuf])
            )
            frames = self.step * self.args.num_workers
            self.logger.log(
                "Training statistics for step: {}".format(frames)
            )
            self.logger.scalar_summary(
                self.info.avg,
                self.step * self.args.num_workers,
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