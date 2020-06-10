import os
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gym
from agents.a2c.network import ActorCritic
import settings
import envs
from utils.logger import Logger
from utils.summary import EvaluationMetrics
from utils.common import load_model, safemean

__all__ = ['A2CAgent']


class A2CAgent:
    def __init__(self, args, name='A2C'):
        self.args = args
        self.env = getattr(envs, args.env)(args)
        self.disc = type(self.env.action_space) != gym.spaces.box.Box
        if args.env == 'atari':
            self.frame_count = 1
        else:
            self.frame_count = 4
        self.logger = Logger(name, args=args)
        self.step = 0
        self.log_step = args.log_step // args.num_workers
        if args.save_step is None:
            self.save_step = None
        else:
            self.save_step = args.save_step // args.num_workers
        self.video_flag = False
        self.record_flag = False

        # Define constants
        self.gam = self.args.gam
        self.lam = self.args.lam
        self.ent_coef = self.args.ent_coef
        self.vf_coef = self.args.vf_coef
        self.max_grad = 0.5
        self.update_step = args.n_step
        assert self.update_step * args.num_workers >= args.batch_size

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
            else:
                n_actions = len(self.env.action_space.sample())
            model = ActorCritic(input_shape, n_actions, disc=self.disc)

        model = model.to(args.device)
        self.policy = model
        self.keys = ['obs', 'acs', 'vals', 'nlps', 'rews', 'dones']
        self.buffer = {k: [] for k in self.keys}

        # Create environment
        self.obs = self.env.reset()
        self.epinfobuf = deque(maxlen=100)
        dones = [False for _ in range(args.num_workers)]
        self.dones = np.asarray(dones, dtype=bool)
        self.imgs = [self.env.render(mode='rgb_array')]

        # Define optimizer
        self.optim = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=args.lr,
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

    def collect(self, **kwargs):
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            ac_dist, val_dist = self.policy(obs, **kwargs)
            acs = ac_dist.sample()
            nlps = -ac_dist.log_prob(acs)
            vals = val_dist.mean
            ents = ac_dist.entropy().squeeze()
        obs, acs, vals, nlps, ents = map(
            lambda x: x.cpu().numpy(),
            [obs, acs, vals, nlps, ents]
        )
        dones = np.asarray(self.dones, dtype=np.float)

        self.buffer['obs'].append(obs)
        self.buffer['acs'].append(acs)
        self.buffer['vals'].append(vals)
        self.buffer['nlps'].append(nlps)
        self.buffer['dones'].append(dones)

        self.obs, rews, self.dones, infos = self.env.step(acs)
        if self.video_flag:
            self.video_summary()

        self.dones = self.dones.astype(np.float)
        self.info.update('Values/Reward', rews.mean().item())
        self.buffer['rews'].append(rews)
        for info in infos:
            if 'episode' in info:
                self.epinfobuf.append(info['episode']['score'])
        return ac_dist, val_dist

    def video_summary(self):
        if self.dones[0]:
            self.record_flag = True
            self.counter = 0
            if len(self.imgs) > 0:
                imgs = np.asarray(self.imgs)
                self.logger.video_summary(
                    imgs,
                    self.step*self.args.num_workers,
                    tag='Playback',
                )
                self.video_flag = False
                self.record_flag = False
            self.imgs = []
        elif self.record_flag:
            self.counter += 1
            if self.counter == self.frame_count:
                self.imgs.append(
                    self.env.render(mode='rgb_array').astype(np.uint8))
                self.counter = 0

    def compute_advantage(self, **kwargs):
        # General Advantage Estimation
        gae = 0
        advs = torch.zeros_like(self.buffer['vals'])
        dones = torch.from_numpy(self.dones).to(self.args.device)

        # Look one more frame further
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            _, val_dist = self.policy(obs, **kwargs)
            vals = val_dist.mean

        for t in reversed(range(self.update_step)):
            if t == self.update_step - 1:
                _vals = vals
                _nonterminal = 1.0 - dones.float()
            else:
                _vals = self.buffer['vals'][t + 1]
                _nonterminal = 1.0 - self.buffer['dones'][t + 1].float()

            while len(_nonterminal.shape) < len(_vals.shape):
                _nonterminal = _nonterminal.unsqueeze(1)
            rews = self.buffer['rews'][t]
            while len(rews.shape) < len(_vals.shape):
                rews = rews.unsqueeze(1)

            vals = self.buffer['vals'][t]
            delta = rews + _nonterminal * self.gam * _vals - vals
            gae = delta + _nonterminal * self.gam * self.lam * gae
            advs[t] = gae

        return advs.detach()

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

        advs = self.buffer['advs'][idx]
        rets = self.buffer['vals'][idx] + advs
        self.info.update('Values/Adv', advs.max().item())

        vf_loss = (rets.detach() - vals).pow(2).mean()
        self.info.update('Loss/Value', vf_loss.item())

        # Policy gradient according to advantage
        pg_loss = (advs.detach() * nlps.unsqueeze(-1)).mean()
        self.info.update('Loss/Policy', pg_loss.item())

        # Total loss
        loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss
        self.info.update('Loss/Total', loss.item())
        return loss

    def train(self, **kwargs):
        self.policy.train()
        st = time.time()
        self.step += 1

        self.collect(**kwargs)
        if self.step % self.update_step == 0:
            for k, v in self.buffer.items():
                v = torch.from_numpy(np.asarray(v))
                self.buffer[k] = v.float().to(self.args.device)
            self.buffer['advs'] = self.compute_advantage(**kwargs)
            for k, v in self.buffer.items():
                self.buffer[k] = v.view(-1, *v.shape[2:])

            for _ in range(self.args.epoch):
                # Shuffle collected batch
                idx = np.arange(len(self.buffer['advs']))
                np.random.shuffle(idx)

                # Train from collected batch
                start = 0
                batch_size = self.args.batch_size
                for _ in range(len(self.buffer['advs']) // batch_size):
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
        self.info.update('Time/Item', elapsed / self.args.num_workers)
        if self.step % self.log_step == 0:
            self.video_flag = self.args.playback
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
            self.info.reset()
