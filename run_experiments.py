"""Automatically check and run experiments in queue, in synchronous fashion"""
import os
import itertools
import argparse
import random
import numpy as np
import torch
import time
from multiprocessing import Process
import psutil
import GPUtil
import yaml
import json
from settings import PROJECT_ROOT
from utils.logger import Logger


def run_experiment(seed, gpu_id, env_id, agent_name):
    path = os.path.join(PROJECT_ROOT, 'experiments', 'config.json')
    with open(path, 'r+') as config:
        args = json.load(config)
        args['tag'] = "../../../experiments/" + agent_name + '/' + env_id
        args['mode'] = agent_name
        args['seed'] = seed
        args['debug'] = True
        args['steps'] = 5e7
        args['env_id'] = env_id
        args['device'] = 'cuda:{}'.format(gpu_id)

        config.seek(0)
        json.dump(args, config)
        config.truncate()

    cmd = "python main.py --load_config {}".format(path)
    os.system("nohup {} >/dev/null 2>&1 &".format(cmd))


def run_loop(tasks, args):
    args.log_level = 20
    args.tag = "../experiments"
    logger = Logger('MAIN', args)

    gpu_recent = {k: 0.0 for k in args.gpus}
    while True:
        # Check CPU usage
        cpu_usage = psutil.cpu_percent(interval=args.interval, percpu=True)
        cpu_busy = all([p > args.max_cpu * 100 for p in cpu_usage])

        # Check RAM usage
        ram_usage = psutil.virtual_memory().percent
        ram_busy = ram_usage > args.max_ram * 100

        if cpu_busy:
            logger.log('CPU is busy', 'DEBUG')
        elif ram_busy:
            logger.log('RAM is busy', 'DEBUG')

        if not (cpu_busy or ram_busy):
            # Check VRAM usage
            found = False
            gpus = GPUtil.getGPUs()
            for gpu_id in args.gpus:
                if gpus[gpu_id].memoryFree > args.min_gpu:
                    diff = time.time() - gpu_recent[gpu_id]
                    if diff > args.gpu_interval:
                        env_id = tasks['To Do'].pop()
                        tasks['Done'].append(env_id)
                        run_experiment(args.seed, gpu_id, env_id, args.agent)
                        gpu_recent[gpu_id] = time.time()
                        found = True

                        n_done = len(tasks['Done'])
                        n_total = n_done + len(tasks['To Do'])
                        msg = "({}/{}) run {} for {} on gpu {}"
                        logger.log(msg.format(n_done, n_total, env_id, args.agent, gpu_id))
                        break
            if not found:
                logger.log('GPU is busy', 'DEBUG')

        # Update status file
        path = os.path.join(PROJECT_ROOT, 'experiments', args.agent)
        status = os.path.join(path, 'status.yaml')
        with open(status, 'w') as f:
            yaml.dump(tasks, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Options for running the full suite for Atari"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--agent", type=str, default=None,
                        choices=['ppo', 'iqpg', 'iqpg_energy', 'mogpg'])
    parser.add_argument("--gpus", nargs='+', type=int)
    parser.add_argument("--debug", "-d", action='store_true')

    parser.add_argument_group("monitoring options")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--gpu_interval", type=float, default=300)
    parser.add_argument("--max_cpu", type=float, default=0.9,
                        help="maximum CPU utilization rate")
    parser.add_argument("--max_ram", type=float, default=0.9,
                        help="maximum RAM utilization rate")
    parser.add_argument("--min_gpu", type=float, default=3000,
                        help="minimum GPU utilization in Mb")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create status file
    assert args.agent is not None
    path = os.path.join(PROJECT_ROOT, 'experiments', args.agent)
    os.makedirs(path, exist_ok=True)
    status = os.path.join(path, 'status.yaml')
    if args.reset or not os.path.isfile(status):
        atari = os.path.join(PROJECT_ROOT, 'experiments', 'atari62.txt')
        with open(atari) as f:
            games = f.read().split('\n')
        dict_file = {
            'To Do': games,
            'Done': []
        }
        with open(status, 'w') as f:
            yaml.dump(dict_file, f)

    with open(status) as f:
        tasks = yaml.full_load(f)

    args.log_level = 1 if args.debug else 20
    args.tag = "../experiments"

    # Start main loop
    run_loop(tasks, args)
