import math
import torch


def _sqdiff(mu1, sig1, w1, mu2, sig2, w2):
    _mu = torch.abs(mu1.unsqueeze(1) - mu2.unsqueeze(2))
    _var = sig1.unsqueeze(1).pow(2) + sig2.unsqueeze(2).pow(2)
    mu = (torch.sqrt(_var * 2 / math.pi)
          * torch.exp(-_mu.pow(2) / (2 * _var + 1e-8))
          + _mu * torch.erf(_mu / (torch.sqrt(2 * _var) + 1e-8)))
    d = mu
    summ = w1.unsqueeze(1) * w2.unsqueeze(2) * d
    return summ.sum(-1).sum(-1, keepdim=True)


def Cramer(mu1, sig1, w1, mu2, sig2, w2):
    mu2 = mu2.detach()
    sig2 = sig2.detach()
    w2 = w2.detach()
    loss = (2 * _sqdiff(mu1, sig1, w1, mu2, sig2, w2)
            - _sqdiff(mu1, sig1, w1, mu1, sig1, w1)
            - _sqdiff(mu2, sig2, w2, mu2, sig2, w2))
    return loss


def huber_quantile_loss(input, target, quantiles):
    n_quantiles = quantiles.size(-1)
    diff = target.unsqueeze(2) - input.unsqueeze(1)
    taus = quantiles.unsqueeze(-1).unsqueeze(1)
    taus = taus.expand(-1, n_quantiles, -1, -1)
    loss = diff.pow(2) * (taus - (diff < 0).float()).abs()
    return loss.squeeze(3).sum(-1).mean(-1)


def sample_cramer(samples1, samples2):
    d = (2 * _sqdiff_sample(samples1, samples2)
         - _sqdiff_sample(samples1, samples1)
         - _sqdiff_sample(samples2, samples2))
    return d


def _sqdiff_sample(samples1, samples2):
    assert samples1.size() == samples2.size()
    assert len(samples1.size()) == 2
    diff = samples1.unsqueeze(1) - samples2.unsqueeze(2)
    return diff.abs().mean(-1, keepdim=True)
