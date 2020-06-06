import numpy as np
import torch
import torch.nn as nn
import copy


class ArgumentParser(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class RMS:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = None
        self.var = None
        self.count = 0

    def update(self, x):
        # mean = x.mean(0, keepdims=True)
        # var = x.var(0, keepdims=True)
        mean = x.mean()
        var = x.var()
        count = x.shape[0]
        self._update(mean, var, count)
        return self.stat()

    def _update(self, mean, var, count):
        if self.mean is not None:
            new_count = self.count + count
            delta = mean - self.mean
            new_mean = self.mean + delta * count / new_count
            m_a = self.var * self.count
            m_b = var * count
            M2 = m_a + m_b + delta**2 * self.count * count / new_count
            new_var = M2 / new_count
        else:
            new_mean = mean
            new_var = var
            new_count = count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def stat(self):
        if self.mean is not None:
            return copy.deepcopy(self.mean), copy.deepcopy(self.var)
        else:
            raise ValueError("RMS error: Data has not been provided")


class EMS:
    def __init__(self, gamma):
        self.alpha = 1 - gamma
        self.mean = 0.0
        self.var = 0.0
        self.count = 0.0

    def update(self, x):
        delta = x.mean() - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)
        self.count += len(x)


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = 0.0
        self.gamma = gamma

    def update(self, rews, dones):
        self.rewems = rews + self.gamma * self.rewems
        return self.rewems


class ReturnFilter:
    def __init__(self, gamma, clip=1.0):
        self.ret = 0.0
        self.gamma = gamma
        self.rms = RMS()
        self.clip = clip

    def update(self, x, dones):
        self.ret = self.gamma * self.ret + x
        self.rms.update(self.ret)
        self.ret = (1.0 - dones) * self.ret

    def filter(self, x):
        rew = x / (self.rms.var**0.5 + 1e-6)
        rew = torch.clamp(rew, -self.clip, self.clip)
        return rew


def save_model(model, path):
    if isinstance(model, nn.DataParallel):
        model = model.module
    pth = {
        'model': model,
        'checkpoint': model.state_dict(),
    }
    torch.save(pth, path)


def load_model(path):
    pth = torch.load(path, map_location=lambda storage, loc: storage)
    model = pth['model']
    model.load_state_dict(pth['checkpoint'])
    return model


def safemean(x):
    return np.nan if len(x) == 0 else np.nanmean(x)


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2) + 1))
