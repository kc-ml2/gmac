import os
import time
import numpy as np
import torch
import torch.nn as nn
import settings
from utils.common import save_model, load_model, safemean, ReturnFilter
from agents.a2c.train import A2CAgent
from agents.mogpg.network import MGAC
from agents.acktr.kfac import KFACOptimizer
from torch.distributions import Bernoulli, Normal, Categorical
from utils.loss import Cramer, sample_cramer
from utils.summary import EvaluationMetrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
sns.set(style="white", palette="muted", color_codes=True)


__all__ = ['MGPGAgent', 'M2CAgent', 'MCKTRAgent']


def logmod(x):
    return torch.sign(x) * torch.log(x.abs() + 1.0)


def lt_mean(mu, sig, w, a=None):
    if a is not None:
        pass
    else:
        a = (mu * w).sum(-1, keepdim=True)
    std_dist = Normal(0, 1)
    alpha = (a - mu) / (sig + 1e-8)
    phi = torch.exp(std_dist.log_prob(alpha))
    z = 1.0 - phi
    # tg = torch.sum(w * (z * mu + sig * phi - z * a), dim=-1, keepdim=True)
    tg = torch.sum(w * (mu + sig * phi / z), dim=-1, keepdim=True)
    return tg


def truncated(x, beta=0.5):
    mean_x = x.mean(-1, keepdim=True)
    mask = (x >= mean_x).float()
    mask_sum = mask.sum(-1, keepdim=True)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask), mask_sum)
    masked_mean = (x * mask).sum(-1, keepdim=True) / mask_sum
    # diff = masked_mean - mean_x

    # return mean_x + beta * diff

    var = (x - masked_mean)**2
    masked_var = (var * mask).sum(-1, keepdim=True) / mask_sum
    std = torch.sqrt(masked_var)

    return mean_x + beta * std


class MGPGAgent(A2CAgent):
    def __init__(self, args, name='MGPG'):
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
            model = MGAC(input_shape, n_actions, disc=self.disc, ac=ac_space,
                         min_var=self.min_sig**2)

        model = model.to(args.device)
        self.policy = model
        if hasattr(model, 'ivalue_head'):
            self.intrinsic = True
            self.keys.append('irews')
            self.buffer['irews'] = []
            self.i_rms = ReturnFilter(self.gam, clip=5.0)
        else:
            self.intrinsic = False

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
        if hasattr(self.buffer['vdists'][0], 'auxs'):
            intrinsic = True
            igae = 0
            # iadvs = torch.zeros_like(self.buffer['vdists'][0].auxs[0].mean)
            _ivals = val_dist.auxs[0].mean
            irews = torch.exp(val_dist.entropy())
        else:
            intrinsic = False
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
            # Maximum entropy learning
            # ent = 0.01 * self.buffer['vdists'][t].entropy()
            # rews += ent

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
            # probs = (pi / (1 + pi))
            dist = Bernoulli(probs=probs)
            mask = dist.sample((self.n_sample,)).to(vals)
            mask = mask.transpose(0, 1).squeeze(-1)
            next_mus += (1 - mask) * (curr_mus - next_mus)
            next_sigs += (1 - mask) * (curr_sigs - next_sigs)

            vals = self.buffer['vals'][t]
            # rews = rews + 0.001 * self.buffer['acdists'][t].entropy()
            delta = rews + _nont * self.gam * _vals - vals
            gae = delta + _nont * self.gam * self.lam * gae
            advs[t] = gae

            if intrinsic:
                ivals = self.buffer['vdists'][t].auxs[0].mean
                irews = self.i_rms.filter(irews) * _nont
                delta = irews + _nont * self.gam * _ivals - ivals
                igae = delta + _nont * self.gam * self.lam * igae
                advs[t] = advs[t] + 0.5 * igae
                irews = torch.exp(self.buffer['vdists'][t].entropy())
                _ivals = ivals

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

        # rets = Normal(self.buffer['ret_mus'][idx],
        #               self.buffer['ret_sigs'][idx]).sample()
        _rets = self.buffer['rets'][idx]
        # rets = truncated(_rets, beta=0.5)
        use_adv = True
        if use_adv:
            advs = self.buffer['advs'][idx]
        else:
            ltm = lt_mean(
                self.buffer['ret_mus'][idx],
                torch.clamp(self.buffer['ret_sigs'][idx], min=self.min_sig),
                1.0 / self.n_sample * torch.ones_like(_rets))
            # tar = self.buffer['ret_mus'][idx].mean(-1, keepdim=True)
            src = self.buffer['vals'][idx]
            advs = (ltm - src).detach()
        # advs = logmod(advs).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)
        self.info.update('Values/Adv', advs.max().item())

        vf_loss = Cramer(val_dist.loc,
                         val_dist.scale,
                         val_dist.w,
                         self.buffer['ret_mus'][idx],
                         self.buffer['ret_sigs'][idx],
                         1 / self.n_sample * torch.ones_like(_rets)).mean()
        #                  torch.clamp(self.buffer['ret_sigs'][idx],
        #                              min=self.min_sig),
        # vf_loss = sample_cramer(val_dist.rsample(self.n_sample),
        #                         _rets).mean()
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


class M2CAgent(MGPGAgent):
    def __init__(self, args, name='M2C'):
        super().__init__(args, name)
        # Define optimizer
        self.optim = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=args.lr
        )

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
        self.info.update('Values/Adv', advs.max().item())

        vf_loss = Cramer(
            val_dist.loc,
            val_dist.scale,
            val_dist.w,
            self.buffer['ret_mus'][idx],
            self.buffer['ret_sigs'][idx],
            1 / self.n_sample * torch.ones_like(_rets)
        ).mean()
        self.info.update('Loss/Value', vf_loss.item())

        # Policy gradient according to advantage
        pg_loss = (advs.detach() * nlps.unsqueeze(-1)).mean()
        self.info.update('Loss/Policy', pg_loss.item())

        # Total loss
        loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss
        self.info.update('Loss/Total', loss.item())
        self.graph_dict = {
            'vals': vals[:20].detach().cpu().numpy(),
            'rets': _rets[:20].detach().cpu().numpy(),
        }
        return loss


class MCKTRAgent(MGPGAgent):
    def __init__(self, args, name='MCKTR'):
        super().__init__(args, name)
        # Define optimizer
        self.optim = KFACOptimizer(
            self.policy,
            lr=args.lr
        )

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
        self.info.update('Values/Adv', advs.max().item())

        vf_loss = Cramer(
            val_dist.loc,
            val_dist.scale,
            val_dist.w,
            self.buffer['ret_mus'][idx],
            self.buffer['ret_sigs'][idx],
            1 / self.n_sample * torch.ones_like(_rets)
        ).mean()
        self.info.update('Loss/Value', vf_loss.item())

        # Policy gradient according to advantage
        pg_loss = (advs.detach() * nlps.unsqueeze(-1)).mean()
        self.info.update('Loss/Policy', pg_loss.item())

        # Compute gradient from Fisher matrix
        if self.optim.steps % self.optim.TCov == 0:
            self.policy.zero_grad()
            pg_fisher = -nlps.mean()
            vf_fisher = -Cramer(
                val_dist.loc,
                val_dist.scale,
                val_dist.w,
                vals,
                torch.ones_like(vals) * 5e-2,
                1 / vals.size(1) * torch.ones_like(vals)
            ).mean()
            fisher_loss = pg_fisher + vf_fisher

            self.optim.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optim.acc_stats = False

        # Total loss
        loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss
        self.info.update('Loss/Total', loss.item())
        self.graph_dict = {
            'vals': vals[:20].detach().cpu().numpy(),
            'rets': _rets[:20].detach().cpu().numpy(),
        }

        return loss