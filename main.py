import os
import gc
import json
from distutils.dir_util import copy_tree
import argparse
import warnings
import random
import numpy as np
import torch
import settings
import utils.common as common
import defaults

warnings.simplefilter('ignore', UserWarning)


def train(args, agent_name, agent_src, train_fn=None):
    """Template function for training various agents.
    """
    import agents
    # Create agent and save current state
    agent = getattr(agents, agent_name)(args)
    logger = agent.logger
    log_dir = logger.log_dir

    path = os.path.join(log_dir, 'config.json')
    logger.log("Saving current arguments to {}".format(path))
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    src = os.path.join(settings.PROJECT_ROOT, 'agents', agent_src)
    dst = os.path.join(log_dir, 'src')
    logger.log("Saving relevant source code to {}".format(dst))
    os.makedirs(dst)
    copy_tree(src, dst)

    # Begin training
    env = "{}:{}".format(args.env, args.env_id)
    logger.log("Begin training {} in ".format(agent_name) + env)
    steps = args.steps // args.num_workers
    for step in range(steps):
        if train_fn is None:
            agent.train()
        else:
            train_fn(agent, step, steps)
        gc.collect()
    logger.log("Finished training {} in ".format(agent_name) + env)


def a2c(args):
    train(args, 'A2CAgent', 'a2c')


def ppo(args):
    train(args, 'PPOAgent', 'ppo')


def iqpg(args):
    train(args, 'IQACAgent', 'iqac')


def iqpg_energy(args):
    train(args, 'IQACEAgent', 'iqace')


def mogpg(args):
    train(args, 'GMACAgent', 'gmac')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributional Perspective on Actor-Critic"
    )
    parser.add_argument("--load_config", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument_group("logger options")
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--log_step", type=float, default=2e4)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--playback", "-p", action="store_true")
    parser.add_argument("--save_step", type=float, default=None)

    parser.add_argument_group("dataset options")
    parser.add_argument("--env", type=str, default='atari')
    parser.add_argument("--env_id", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_step", type=int, default=128)
    parser.add_argument("--exp", "-e", action="store_true")

    parser.add_argument_group("training options")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--steps", type=float, default=5e7)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gam", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--cliprange", type=float, default=0.1)

    args = parser.parse_args()
    if args.load_config is not None:
        path = os.path.join(settings.PROJECT_ROOT, args.load_config)
        with open(path) as config:
            args = common.ArgumentParser(json.load(config))

    if args.exp:
        if args.env == 'atari' or args.env == 'bullet':
            config = args.env
            default_values = getattr(defaults, config)()
            for k, v in default_values.items():
                setattr(args, k, v)

    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.tag is None:
        args.tag = args.mode + '/' + args.env_id
    else:
        args.tag = args.mode + '/' + args.env_id + '/' + args.tag
    args.tag = args.tag.lower()

    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30

    args.steps = int(args.steps)
    args.log_step = int(args.log_step)
    if not hasattr(args, 'device'):
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)
