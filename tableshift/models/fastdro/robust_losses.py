"""
PyTorch modules for computing robust losses with
for (KL-regularized) CVaR, constrained-chi^2 and penalized-chi^2
uncertainty sets.
Includes losses appropriate for our porposed batch and MLMC gradient estimators
as well as losses for the dual-SGM and primal-dual methods.

Code from https://github.com/daniellevy/fast-dro
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from tableshift.models.fastdro.utils import project_to_cs_ball, project_to_cvar_ball
from tableshift.models.fastdro.datasets import CustomDistributionSampler


GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5


def chi_square_value(p, v, reg):
    """Returns <p, v> - reg * chi^2(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        chi2 = (0.5 / m) * reg * (
                torch.norm(m * p - torch.ones(m, device=v.device), p=2) ** 2)

    return torch.dot(p, v) - chi2


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def fenchel_kl_cvar(v, alpha):
    """Returns the empirical mean of the Fenchel dual for KL CVaR"""
    v -= np.log(1 / alpha)
    v1 = v[torch.lt(v, 0)]
    v2 = v[torch.ge(v, 0)]
    w1 = torch.exp(v1) / alpha - 1
    w2 = (v2 + 1) * (1 / alpha) - 1
    return (w1.sum() + w2.sum()) / v.shape[0]


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max] 
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the 
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    logging.warning('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)


def huber_loss(x, delta=1.):
    """ Standard Huber loss of parameter delta

    https://en.wikipedia.org/wiki/Huber_loss

    returns 0.5 * x^2 if |a| <= \delta
            \delta * (|a| - 0.5 * \delta) o.w.
    """
    if torch.abs(x) <= delta:
        return 0.5 * (x ** 2)
    else:
        return delta * (torch.abs(x) - 0.5 * delta)


class RobustLoss(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, size, reg, geometry, tol=1e-4,
                 max_iter=1000, debugging=False):
        """
        Parameters
        ----------

        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection

        """
        super().__init__()
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging

        self.is_erm = size == 0

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v):
        size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x),
                                     torch.Tensor([1 / size]).type(x.dtype)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return torch.ones_like(v) / m

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return torch.relu(v - eta) / (reg * m)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
        return p(eta_star)

    def forward(self, v):
        """Value of the robust loss

        Note that the best response is computed without gradients

        Parameters
        ----------

        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples

        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """
        if self.is_erm:
            return v.mean()
        else:
            with torch.no_grad():
                p = self.best_response(v)

            if self.geometry == 'cvar':
                return cvar_value(p, v, self.reg)
            elif self.geometry == 'chi-square':
                return chi_square_value(p, v, self.reg)


class DualRobustLoss(torch.nn.Module):
    """Dual formulation of the robust loss, contains trainable parameter eta"""

    def __init__(self, size, reg, geometry, eta_init=0.0):
        """Constructor for the dual robust loss

        Parameters
        ----------

        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            \chi^2 divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        eta_init : float
            Initial value for equality constraint Lagrange multiplier eta

        """
        super().__init__()
        self.eta = torch.nn.Parameter(data=torch.Tensor([eta_init]))
        self.geometry = geometry
        self.size = size
        self.reg = reg

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

    def forward(self, v):
        """Value of the dual loss on the batch of examples

        Parameters
        ----------

        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples

        Returns
        -------
        loss : torch.float
            Value of the dual of the robust loss on the batch of examples
        """
        n = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg == 0:
                return self.eta + torch.relu(v - self.eta).mean() / self.size
            else:
                return self.eta + self.reg * fenchel_kl_cvar(
                    (v - self.eta) / self.reg, self.size)

        elif self.geometry == 'chi-square':
            w = torch.relu(v - self.eta)

            if self.size == float('inf'):
                return ((0.5 / self.reg) * (w ** 2).mean()
                        + 0.5 * self.reg + self.eta)
            else:
                if self.reg == 0:
                    return self.eta + np.sqrt(
                        (1 + 2 * self.size) / n) * torch.norm(w, p=2)
                else:
                    return self.eta + 0.5 * self.reg + huber_loss(
                        torch.norm(w, p=2) / np.sqrt(n * self.reg),
                        delta=np.sqrt(self.reg * (1 + 2 * self.size)))


class PrimalDualRobustLoss(nn.Module):
    """Primal-dual robust loss.

    This loss is build for a somewhat hacky implmentation of a primal-dual
    optimization algorithm where the dual step (i.e., update to to the sampling
    distribution) takes place *inside the forward pass*. The main drawback is
    that this bakes in the (dual of part of the) optimization algorithm into
    the loss, rather than having PyTorch's different optimizers take care
    of the loss as usual. The main advantage is that this way we can
    directly replace standard primal only losses without changing any
    of the rest of the training code"""
    def __init__(self, size, geometry, sampler: CustomDistributionSampler,
                 step_size=1e-3, clip=0.01):
        super().__init__()
        self.size = size
        self.geometry = geometry
        self.step_size = step_size
        self.clip = clip
        self.is_erm = size == 0

        self.sampler = sampler
        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be <= 1 for cvar, is {self.size}')

    def forward(self, v, update=True):
        if update and not self.is_erm:
            p_update = np.zeros_like(self.sampler.p)
            coefs = self.step_size / (
                    len(v) * self.sampler.p[self.sampler.inds])
            # pdb.set_trace()
            np.add.at(p_update, self.sampler.inds,
                      v.detach().cpu().numpy() * coefs)
#             pdb.set_trace()
            if self.clip is not None:
                p_update = np.minimum(p_update, self.clip)
            # pdb.set_trace()
            if self.geometry == 'chi-square':
                self.sampler.p = project_to_cs_ball(
                    self.sampler.p + p_update, self.size)
            elif self.geometry == 'cvar':
                self.sampler.p = project_to_cvar_ball(
                    self.sampler.p * np.exp(p_update), self.size)

        return v.mean()


class MultiLevelRobustLoss(torch.nn.Module):
    """PyTorch module for multi-level estimator"""

    def __init__(self, robust_loss, batch_size_pmf, batch_size_min):
        """
        Parameters
        ----------

        robust_loss : instance of RobustLoss
            Robust loss to estimate via multi-level
        batch_size_pmf : callable
            int -> float that maps a batch size to its probability
        batch_size_min : int
            Minimum batch size (must be even)

        """
        super().__init__()
        self.robust_loss = robust_loss
        self.batch_size_pmf = batch_size_pmf
        self.batch_size_min = batch_size_min

    def forward(self, v):
        """Estimator of the loss (and gradients) via multi-level

        Parameters
        ----------

        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples

        Returns
        -------
        loss : torch.float
            Returns multi-level estimator of the loss values and its
            gradient is the multi-level estimator of the gradients
        """
        n = v.shape[0]

        half_min_batch = self.batch_size_min // 2
        avg_term = self.robust_loss(v[:half_min_batch])

        delta = (self.robust_loss(v)
                 - 0.5 * (self.robust_loss(v[:n // 2])
                          + self.robust_loss(v[n // 2:])))

        return avg_term + (1.0 / self.batch_size_pmf(n)) * delta
