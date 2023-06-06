"""
Utility functions used throughout the code.

Code from https://github.com/daniellevy/fast-dro
"""

import io
import json
import os
import pickle

import logging

import numpy as np
import pathlib

import torch
from torch.nn import Sequential, Module, Linear
from scipy.sparse import csc_matrix
from scipy import optimize, interpolate
from scipy.stats import norm as Gaussian
from scipy.special import betaln

cifar10_label_names = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck']


def aggregate_by_group(v, g, n):
    assert g.max() < n
    return csc_matrix((v, (g, np.zeros_like(g))), shape=(n, 1)
                      ).toarray().squeeze()


def subsample_arrays(arrays, n, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    take_inds = np.random.choice(len(arrays[0]),
                                 n, replace=False)
    np.random.set_state(rng_state)

    return [a[take_inds] for a in arrays]


def get_weights_norm(model):
    """Returns the average 2-norm of the weights"""
    return np.mean([torch.norm(p.data).item() for p in model.parameters()])


def copy_state(model_src, model_tgt):
    """Copy weights of model_src in model_tgt"""
    model_tgt.load_state_dict(model_src.state_dict())
    return model_tgt


def average_step(model, model_avg, step, eta=0.):
    """In place averaging step from 
    http://proceedings.mlr.press/v28/shamir13.pdf
    
    Parameters
    ----------

    model : torch.Module
        Current model that we optimize
    model_avg : torch.Module
        Model corresponding to the averaging of the iterates
    step : int
        Current iteration number (starts at 1)
    eta : float, optional
        Parameter of [Shamir & Zhang, 2013], eta=0. corresponds to normal
        averaging

    Returns
    -------
    model_avg : torch.Module
        Updated averaged model
    """

    keys = model.state_dict().keys()

    for k in keys:
        model_avg.state_dict()[k].mul_(1 - ((eta + 1) / (step + eta))).add_(
            model.state_dict()[k].mul((eta + 1) / (step + eta)))
    return model_avg


def average_step_ema(model, model_avg, gamma=0.9):
    """Updates model_avg with an exponential moving average with param gamma"""
    keys = model.state_dict().keys()

    for k in keys:
        model_avg.state_dict()[k].mul_(1 - gamma).add_(
            model.state_dict()[k].mul(gamma))
    return model_avg


class SquaredGaussian(object):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale if scale > 0 else 1e-12
        self.gaussian = Gaussian(loc=loc, scale=self.scale)

    def ppf(self, x):
        def target(a):
            return self.cdf(a) - x

        v_hi = self.scale ** 2
        while target(v_hi) < 0:
            v_hi *= 2
        return optimize.brentq(target, 0, v_hi)

    def cdf(self, x):
        return self.gaussian.cdf(np.sqrt(x)) - self.gaussian.cdf(-np.sqrt(x))


def analytical_dro_objective(p, invcdf, size=1.0, geometry='cvar', reg=0.0,
                             output_lambda=False):
    if geometry == 'cvar':
        assert reg == 0.0
        opt_lambda = None
        var = interpolate.interp1d(p, invcdf)(1 - size)
        ind = np.where(p > 1 - size)[0][0]
        out = (1 / (p[-1] - 1 + size)) * np.trapz(
            np.concatenate([[var], invcdf[ind:]]),
            np.concatenate([[1 - size], p[ind:]]))
    elif geometry == 'chi-square':
        if size < np.inf:
            assert reg == 0.0

            def chisquare(eta):
                r = np.maximum(invcdf - eta, 0)
                r /= np.trapz(r, p)
                return 0.5 * np.trapz((r - 1.0) ** 2, p) - size

            eta_0 = invcdf[0]
            while chisquare(eta_0) > 0.0:
                eta_0 = 2 * eta_0 - invcdf[-1]
            eta_1 = (invcdf[-1] + invcdf[-2]) / 2
            if chisquare(eta_1) <= 0.0:
                eta = eta_1
            else:
                eta = optimize.brentq(chisquare, eta_0, eta_1)
            r = np.maximum(invcdf - eta, 0)
            opt_lambda = np.trapz(r, p)
            r /= opt_lambda
            out = np.trapz(r * invcdf, p)
        else:
            assert reg > 0.0

            def target(eta):
                r = np.maximum(invcdf - eta, 0)
                return np.trapz(r, p) / reg - 1.0

            eta_0 = invcdf[0]
            while target(eta_0) < 0.0:
                eta_0 = 2 * eta_0 - invcdf[-1]
            eta_1 = (invcdf[-1] + invcdf[-2]) / 2
            if target(eta_1) >= 0.0:
                eta = eta_1
            else:
                eta = optimize.brentq(target, eta_0, eta_1)
            r = np.maximum(invcdf - eta, 0)
            opt_lambda = np.trapz(r, p)  # should be equal to reg!
            r /= opt_lambda
            out = 0.5 * (np.trapz(r * invcdf, p) + eta + reg)

    if output_lambda:
        return out, opt_lambda
    else:
        return out


def project_to_cs_ball(v, rho):
    """Numpy/Scipy projection to chi-square ball of radius rho"""
    n = len(v)
    def cs_div(p):
        return 0.5 * np.mean((n * p - 1)**2)

    # first, check if a simplex projection is within the chi-square ball
    target_simplex = lambda eta: np.sum(np.maximum(v - eta, 0)) - 1.0
    eta_min_simplex = v.min() - 1 / n
    eta_max_simplex = v.max()
    eta_simplex = optimize.brentq(
        target_simplex, eta_min_simplex, eta_max_simplex)
    p_candidate = np.maximum(v - eta_simplex, 0)
    if cs_div(p_candidate) <= rho:
        return p_candidate

    # second, compute a chi-square best response
    def target_cs(eta, return_p=False):
        p = np.maximum(v - eta, 0)
        if p.sum() == 0.0:
            p[np.argmax(v)] = 1.0
        else:
            p /= p.sum()
        err = cs_div(p) - rho
        return p if return_p else err
    eta_max_cs = v.max()
    eta_min_cs = v.min()
    if target_cs(eta_max_cs) <= 0:
        return target_cs(eta_max_cs, return_p=True)
    while target_cs(eta_min_cs) > 0.0:  # find left interval edge for bisection
        eta_min_cs = 2 * eta_min_cs - eta_max_cs
    eta_cs = optimize.brentq(
        target_cs, eta_min_cs, eta_max_cs)
    p_candidate = target_cs(eta_cs, return_p=True)
    assert np.abs(cs_div(p_candidate) - rho) < rho * 1e-2
    return p_candidate


def project_to_cvar_ball(w, alpha):
    if alpha == 1.0:
        return np.ones(n) / n
    n = len(w)
    k = alpha * n
    w = w + 1e-12  # slight padding to avoid numerical issues
    logw = np.log(w) - np.log(w.max())  # offset so maximum value is 0.0

    def target_cvar(nu, return_p=False):
        w_ = np.exp(np.minimum(logw, nu))
        p = w_ / w_.sum()
        return p if return_p else p.max() - 1 / k

    if target_cvar(0.0) <= 0.0:
        p = w / w.sum()
    else:
        nu_max = 0.0
        nu_min = logw.min()
        nu = optimize.brentq(
            target_cvar, nu_min, nu_max)
        p = target_cvar(nu, return_p=True)
    if p.max() > 1 / k * 1.01:
        logging.warning(f'project_to_cvar_ball: Maximum element of p'
                        f' is {p.max()}; supposed to be {1/k}')
    if np.abs(p.sum() - 1.0) > 1e-2:
        logging.warning(f'project_to_cvar_ball: Elements of p sum to'
                        f'{p.sum()} instead of 1.0')
    return p


def binomln(n, k):
    # Assumes binom(n, k) >= 0
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def subsample_cvar_w(n, s, alpha):
    sk_max = np.floor(alpha * s)
    i_s = np.arange(1, n + 1).reshape(-1, 1)
    k_s = np.arange(1 + sk_max).reshape(1, -1)

    log_choices = (binomln(i_s - 1, k_s)
                   + binomln(n - i_s, s - k_s - 1)
                   - binomln(n, s))

    p_bulk = np.sum(np.exp(log_choices[:, :-1]), axis=1)
    p_edge = np.exp(log_choices[:, -1])

    w = 1 / (alpha * s) * p_bulk + (1 - sk_max / (alpha * s)) * p_edge

    nk_max = np.floor(alpha * n)
    w_pop = (1 / (alpha * n) * (i_s <= nk_max).astype('float')
             + (1 - nk_max / (alpha * n)) * (i_s == 1 + nk_max).astype('float'))
    return w, w_pop.squeeze()
