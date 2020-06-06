import gym
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import NoopResetEnv
from baselines.common.atari_wrappers import MaxAndSkipEnv
from baselines.common.atari_wrappers import EpisodicLifeEnv
from baselines.common.atari_wrappers import FireResetEnv
from baselines.common.atari_wrappers import ClipRewardEnv
from baselines.common.atari_wrappers import WarpFrame
from baselines.common.atari_wrappers import FrameStack
from envs.gym.monitor import Monitor
from envs.gym.vec_env import SubprocVecEnv, DummyVecEnv

__all__ = ['atari', 'atari_explore']


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


def make_atari(env_id, n_env, seed):
    def _make_atari(env_id, subrank=0, seed=None):
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if isinstance(env.observation_space, gym.spaces.Dict):
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(env, allow_early_resets=True)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, 4)
        return env

    def make_thunk(rank):
        return lambda: _make_atari(
            env_id=env_id,
            subrank=rank,
            seed=seed
        )

    set_global_seeds(seed)
    if n_env > 1:
        return SubprocVecEnv([make_thunk(i) for i in range(n_env)])
    else:
        return DummyVecEnv([make_thunk(0)])


def make_atari_explore(env_id, n_env, seed):
    def _make_atari_explore(env_id, subrank=0, seed=None):
        env = gym.make(env_id)
        env._max_episode_steps = 4500 * 4
        env.spec.timestep_limit = 4500 * 4
        env = StickyActionEnv(env)
        env = MaxAndSkipEnv(env, skip=4)
        if isinstance(env.observation_space, gym.spaces.Dict):
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(env, allow_early_resets=True)
        # env = EpisodicLifeEnv(env)  # XXX: TO BE REMOVED
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, 4)
        return env

    def make_thunk(rank):
        return lambda: _make_atari_explore(
            env_id=env_id,
            subrank=rank,
            seed=seed
        )

    set_global_seeds(seed)
    if n_env > 1:
        return SubprocVecEnv([make_thunk(i) for i in range(n_env)])
    else:
        return DummyVecEnv([make_thunk(0)])


def atari(args):
    return make_atari(
        args.env_id + "NoFrameskip-v4",
        args.num_workers,
        args.seed
    )


def atari_explore(args):
    return make_atari_explore(
        args.env_id + "NoFrameskip-v4",
        args.num_workers,
        args.seed
    )
