"""
methods.py
----------
Numerical optimization methods for MLE estimation of Normal distribution.
Implemented from scratch for educational and portfolio purposes.

Methods:
    - log_likelihood     : compute log-likelihood of Normal distribution
    - newton_mle_mu      : Newton's method to estimate mu
    - bisection_mle_mu   : Bisection method to estimate mu
    - golden_section_mle : Golden Section search (works for mu or sigma)

Author: Aishwarya Lakshmi S
Project: MLE Optimizer
"""

import numpy as np


def log_likelihood(mu, sigma, data):
    """
    Compute log-likelihood of Normal distribution.

    Parameters
    ----------
    mu    : float — mean parameter
    sigma : float — standard deviation (must be > 0)
    data  : array — observed data

    Returns
    -------
    float — log-likelihood value
    """
    n = len(data)
    ll = -n/2 * np.log(2 * np.pi * sigma**2) \
         - (1 / (2 * sigma**2)) * np.sum((data - mu)**2)
    return ll


def newton_mle_mu(data, mu_init=0.0, tol=1e-6, max_iter=100):
    """
    Newton's method to find MLE of mu for Normal distribution.

    Uses first and second derivatives of log-likelihood w.r.t mu:
        d1 = (1/sigma^2) * sum(xi - mu)
        d2 = -n / sigma^2

    Parameters
    ----------
    data     : array — observed data
    mu_init  : float — initial guess for mu (default 0.0)
    tol      : float — convergence tolerance (default 1e-6)
    max_iter : int   — maximum iterations (default 100)

    Returns
    -------
    mu_hat  : float — MLE estimate of mu
    history : list  — [(iteration, mu_estimate, log_likelihood), ...]
    """
    n = len(data)
    sigma = np.std(data)
    mu = mu_init
    history = []

    for i in range(max_iter):
        d1 = (1 / sigma**2) * np.sum(data - mu)   # first derivative
        d2 = -n / sigma**2                          # second derivative
        mu_new = mu - d1 / d2                       # Newton update

        ll = log_likelihood(mu_new, sigma, data)
        history.append((i + 1, mu_new, ll))

        if abs(mu_new - mu) < tol:
            break

        mu = mu_new

    return mu_new, history


def bisection_mle_mu(data, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find MLE of mu by solving dL/dmu = 0.

    Parameters
    ----------
    data     : array — observed data
    a, b     : float — interval [a, b] where derivative changes sign
    tol      : float — convergence tolerance (default 1e-6)
    max_iter : int   — maximum iterations (default 100)

    Returns
    -------
    mu_hat  : float — MLE estimate of mu
    history : list  — [(iteration, mu_estimate, log_likelihood), ...]
    """
    sigma = np.std(data)
    history = []

    def derivative(mu):
        return (1 / sigma**2) * np.sum(data - mu)

    if derivative(a) * derivative(b) > 0:
        raise ValueError("No sign change in interval. Choose a wider interval.")

    for i in range(max_iter):
        mid = (a + b) / 2
        ll = log_likelihood(mid, sigma, data)
        history.append((i + 1, mid, ll))

        if abs(derivative(mid)) < tol or (b - a) / 2 < tol:
            break

        if derivative(a) * derivative(mid) < 0:
            b = mid
        else:
            a = mid

    return mid, history


def golden_section_mle(data, a, b, param='mu', tol=1e-6, max_iter=100):
    """
    Golden Section Search to find MLE of mu or sigma.
    Does not require any derivatives — only log-likelihood values.

    Parameters
    ----------
    data  : array  — observed data
    a, b  : float  — search interval
    param : str    — 'mu' or 'sigma' (which parameter to estimate)
    tol   : float  — convergence tolerance (default 1e-6)
    max_iter : int — maximum iterations (default 100)

    Returns
    -------
    param_hat : float — MLE estimate
    history   : list  — [(iteration, param_estimate, log_likelihood), ...]
    """
    gr = (np.sqrt(5) - 1) / 2   # golden ratio ≈ 0.618
    history = []

    # Fix the other parameter at its sample estimate
    mu_fixed    = np.mean(data)
    sigma_fixed = np.std(data)

    def objective(val):
        if param == 'mu':
            return log_likelihood(val, sigma_fixed, data)
        else:
            return log_likelihood(mu_fixed, val, data)

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = objective(c), objective(d)

    for i in range(max_iter):
        mid = (a + b) / 2
        history.append((i + 1, mid, objective(mid)))

        if abs(b - a) < tol:
            break

        if fc < fd:
            a = c
            c = d;  fc = fd
            d = a + gr * (b - a)
            fd = objective(d)
        else:
            b = d
            d = c;  fd = fc
            c = b - gr * (b - a)
            fc = objective(c)

    return (a + b) / 2, history
