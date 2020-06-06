import os
import contextlib
import gym
import pybullet_envs
import numpy as np
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
# from baselines.common.atari_wrappers import FrameStack
from envs.pybullet.monitor import Monitor
from envs.pybullet.vec_env import SubprocVecEnv, DummyVecEnv
from envs.pybullet.shmem_vec_env import ShmemVecEnv
from utils.common import RMS
import settings

__all__ = ['bullet', 'bullet_raw']


class StateRewNorm:
    def __init__(self, env):
        # super().__init__(env)
        self.__dict__ = env.__dict__.copy()
        self.env = env
        self.obs_rms = RMS()
        self.rew_rms = RMS()
        self.clipob = 10.0
        self.cliprew = 10.0
        self.ret = 0.0
        self.gamma = 0.99

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.ret = self.ret * self.gamma + rew
        obs_mean, obs_var = self.obs_rms.update(obs)
        rew_mean, rew_var = self.rew_rms.update(self.ret)
        obs = np.clip((obs - obs_mean) / (np.sqrt(obs_var) + settings.EPS),
                      -self.clipob, self.clipob)
        rew = np.clip((rew - rew_mean) / (np.sqrt(rew_var) + settings.EPS),
                      -self.cliprew, self.cliprew)
        self.ret *= (1.0 - done)
        return obs, rew, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)


class SkipEnv(gym.Wrapper):
    def __init__(self, env, n_skip=4):
        super().__init__(env)
        self.skip = n_skip

    def step(self, action):
        sum_rew = 0
        for _ in range(self.skip):
            obs, rew, done, info = self.env.step(action)
            sum_rew += rew
            if done:
                break
        return obs, sum_rew, done, info


class LinearStack(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.stack = n_stack
        self.buf = deque(maxlen=self.stack)
        for _ in range(self.stack):
            self.buf.append(np.zeros(self.env.observation_space.shape))
        env_obs = env.observation_space
        low = np.repeat(env_obs.low, self.stack, axis=0)
        high = np.repeat(env_obs.high, self.stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=env_obs.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack):
            self.buf.append(np.zeros(self.env.observation_space.shape))
        self.buf.append(obs)
        return np.concatenate(list(self.buf))

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.buf.append(obs)
        obs_stack = np.concatenate(list(self.buf))
        return obs_stack, rew, done, info


def make_bullet(env_id, n_env, seed, gam=0.99, norm=True):
    def _make_bullet(env_id, subrank=0, seed=None):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            env = gym.make(env_id)
        if isinstance(env.observation_space, gym.spaces.Dict):
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(env, allow_early_resets=True)
        # env = SkipEnv(env, n_skip=4)
        # env = LinearStack(env, n_stack=4)
        return env

    def make_thunk(rank):
        return lambda: _make_bullet(
            env_id=env_id,
            subrank=rank,
            seed=seed
        )

    set_global_seeds(seed)
    if n_env > 1:
        env = SubprocVecEnv([make_thunk(i) for i in range(n_env)])
        # env = ShmemVecEnv([make_thunk(i) for i in range(n_env)])
    else:
        env = DummyVecEnv([make_thunk(0)])
    if norm:
        env = VecNormalize(env, gamma=gam)
    return env


def bullet(args):
    return make_bullet(
        args.env_id + "BulletEnv-v0",
        args.num_workers,
        args.seed,
        args.gam
    )


def bullet_raw(args):
    return make_bullet(
        args.env_id + "BulletEnv-v0",
        args.num_workers,
        args.seed,
        args.gam,
        norm=False
    )
