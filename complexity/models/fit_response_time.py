#!/usr/bin/env python3
# ============================================================
# Joint + RT-only model suite for COCO-style visual search data
# ============================================================
# Models (M6 removed):
#   RT-only
#     M1  log(RT) ~ Normal
#     M2  log(RT) ~ StudentT
#     M3  RT ~ ShiftedLogNormal (RT = tau + LogNormal)
#     M4  RT ~ ExGaussian
#
#   Joint (adds N model + shared latent image difficulty; RT scored marginalizing N)
#     M5a Joint-Normal   (xN = log(1+N))
#     M5b Joint-Normal   (xN = N)
#     M2J Joint-StudentT (xN = log(1+N))
#     M3J Joint-ShiftedLogNormal (xN = log(1+N))
#     M4J Joint-ExGaussian (xN = log(1+N))
#
# Comparison:
#   - PSIS-LOO on a common target: log(RT) density.
#     For models whose likelihood is specified on RT (M3/M4 and joint variants), we apply
#     Jacobian correction: log p(logRT) = log p(RT) + logRT.
#
# CV kept:
#   - Image-stratified cell-held-out CV (option 2): hold out subject×image cells, ensuring
#     each fold retains training data for each image where possible.
#     Scoring for joint models is marginal over N (does not condition on observed N).
#     Also computes ranking stability metrics across folds (including uncertainty-aware ones).
#
# Automatic model selection:
#   - Builds a single table combining: marginal RT accuracy + ranking stability metrics,
#     and selects the best model by a composite rank-based score.
#
# Default target_accept: 0.95
#
# ============================================================

import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az
import xarray as xr

import pytensor
import pytensor.tensor as pt

# Optional NUMBA mode (sometimes faster; sometimes unstable depending on env)
# if os.environ.get("PYTENSOR_MODE", "").upper() == "NUMBA" or os.environ.get("USE_NUMBA", "0") == "1":
# pytensor.config.cxx = ""
# pytensor.config.mode = "NUMBA"

# SciPy optional
try:
    from scipy.special import logsumexp as _logsumexp
except Exception:
    _logsumexp = None

import math


# -------------------------
# Small numeric utilities
# -------------------------

def logsumexp(a, axis=None):
    if _logsumexp is not None:
        return _logsumexp(a, axis=axis)
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def log_ndtr(z):
    # log Phi(z) using erfc; stable-ish, no SciPy required.
    # Phi(z) = 0.5 * erfc(-z/sqrt(2))
    return np.log(0.5 * np.clip(np.vectorize(math.erfc)(-z / np.sqrt(2.0)), 1e-300, 1.0))


def exgaussian_logpdf(x, mu, sigma, nu):
    """
    ExGaussian logpdf with parameters:
      X = Normal(mu, sigma) + Exponential(scale=nu)
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    nu = np.asarray(nu)
    # formula: f(x)= (1/nu) exp((sigma^2)/(2 nu^2) - (x-mu)/nu) * Phi((x-mu)/sigma - sigma/nu)
    z = (x - mu) / sigma - sigma / nu
    logf = (
        -np.log(nu)
        + (sigma * sigma) / (2.0 * nu * nu)
        - (x - mu) / nu
        + log_ndtr(z)
    )
    return logf


def studentt_logpdf(y, mu, sigma, nu):
    """
    StudentT logpdf for y with location mu, scale sigma, dof nu.
    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    nu = np.asarray(nu)

    # log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5 log(nu*pi) - log sigma
    # - (nu+1)/2 * log(1 + ((y-mu)/sigma)^2 / nu)
    lgamma = np.vectorize(math.lgamma)
    a = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)
    b = -0.5 * np.log(nu * np.pi) - np.log(sigma)
    z2 = ((y - mu) / sigma) ** 2
    c = -0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    return a + b + c


def normal_logpdf(y, mu, sigma):
    y = np.asarray(y)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def shifted_lognormal_logpdf_rt(rt, mu, sigma, tau):
    """
    log p(RT) for RT = tau + LogNormal(mu, sigma).
    """
    rt = np.asarray(rt)
    # x will imply the broadcast shape (S, T)
    x = rt - tau
    
    # Initialize out with the shape of x, not rt
    out = np.full_like(x, -np.inf, dtype=float)
    
    ok = x > 0
    
    # Broadcast parameters to match x explicitly
    mu_bc = np.broadcast_to(mu, x.shape)
    sigma_bc = np.broadcast_to(sigma, x.shape)
    
    # Extract only the valid elements
    x_val = x[ok]
    mu_val = mu_bc[ok]
    sigma_val = sigma_bc[ok]
    
    lx = np.log(x_val)
    
    out[ok] = (
        -lx
        - np.log(sigma_val)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * ((lx - mu_val) / sigma_val) ** 2
    )
    return out


def negbin_gamma_poisson_sample(mu, alpha, rng: np.random.Generator):
    """
    Sample NegBin(mu, alpha) via Gamma-Poisson mixture.
    This supports non-integer alpha robustly.

    Parameterization consistent with PyMC NegativeBinomial(mu=mu, alpha=alpha):
      lambda ~ Gamma(shape=alpha, scale=mu/alpha)
      N ~ Poisson(lambda)
    """
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    lam = rng.gamma(shape=alpha, scale=np.clip(mu / alpha, 1e-12, None))
    return rng.poisson(lam)


# -------------------------
# I/O + checkpointing
# -------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_idata(idata: az.InferenceData, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    idata.to_netcdf(tmp)
    tmp.replace(path)


def load_idata(path: Path) -> az.InferenceData:
    return az.from_netcdf(path)


def update_manifest(manifest_path: Path, manifest: dict, key: str, **fields):
    manifest.setdefault("records", {})
    manifest["records"].setdefault(key, {})
    manifest["records"][key].update(fields)
    manifest["last_updated"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# -------------------------
# Data preparation
# -------------------------

def prepare_df(
    df: pd.DataFrame,
    rt_col="RT",
    n_col="N",
    subj_col="subject",
    img_col="image",
    trial_col="trial",
):
    d = df.copy()

    # required columns
    req = [rt_col, subj_col, img_col, trial_col]
    d = d.dropna(subset=req).copy()
    d[rt_col] = pd.to_numeric(d[rt_col], errors="coerce")
    d = d.dropna(subset=[rt_col]).copy()
    d = d[d[rt_col] > 0].copy()

    d[subj_col] = d[subj_col].astype(str)
    d[img_col] = d[img_col].astype(str)

    d["RT"] = d[rt_col].astype(float)
    d["log_rt"] = np.log(d["RT"].values)

    d["trial"] = pd.to_numeric(d[trial_col], errors="coerce")
    d = d.dropna(subset=["trial"]).copy()
    d["log_trial"] = np.log1p(d["trial"].astype(float).values)

    has_N = n_col in d.columns
    if has_N:
        d[n_col] = pd.to_numeric(d[n_col], errors="coerce")
        d = d.dropna(subset=[n_col]).copy()
        d = d[d[n_col] >= 0].copy()
        d["N"] = d[n_col].astype(int)
        d["logN1p"] = np.log1p(d["N"].astype(float))
    else:
        d["N"] = np.nan
        d["logN1p"] = np.nan

    subj_idx, subj_levels = pd.factorize(d[subj_col], sort=True)
    img_idx, img_levels = pd.factorize(d[img_col], sort=True)
    d["subj_idx"] = subj_idx.astype(int)
    d["img_idx"] = img_idx.astype(int)

    # global centering (fold CV will re-center)
    lt_mean = float(d["log_trial"].mean())
    d["log_trial_c"] = d["log_trial"] - lt_mean

    ln_mean = None
    if has_N and np.isfinite(d["logN1p"]).all():
        ln_mean = float(d["logN1p"].mean())
        d["logN1p_c"] = d["logN1p"] - ln_mean
    else:
        d["logN1p_c"] = np.nan

    meta = dict(
        subjects=subj_levels.tolist(),
        images=img_levels.tolist(),
        has_N=bool(has_N and np.isfinite(d["N"]).all()),
        log_trial_mean=lt_mean,
        logN1p_mean=ln_mean,
    )
    return d.reset_index(drop=True), meta


# -------------------------
# Random effects helper
# -------------------------

def _noncentered_re(name, dims, sigma_prior=0.5):
    sigma = pm.HalfNormal(f"sigma_{name}", sigma_prior)
    z = pm.Normal(f"z_{name}", 0.0, 1.0, dims=dims)
    re = pm.Deterministic(name, z * sigma, dims=dims)
    return re, sigma


# -------------------------
# Model builders (RT-only)
# -------------------------

def build_M1_rt_lognormal(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        y = pm.Data("log_rt_obs", d["log_rt"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 0.0, 1.0)
        b_trial = pm.Normal("b_trial", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]
        pm.Normal("logrt_like", mu=mu, sigma=sigma, observed=y, dims="obs")
    return m


def build_M2_rt_studentt(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        y = pm.Data("log_rt_obs", d["log_rt"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 0.0, 1.0)
        b_trial = pm.Normal("b_trial", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)
        nu = pm.Exponential("nu_minus2_over10", 1 / 10) + 2

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]
        pm.StudentT("logrt_like", nu=nu, mu=mu, sigma=sigma, observed=y, dims="obs")
    return m


def build_M3_rt_shifted_lognormal(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        RT_obs = pm.Data("RT_obs", d["RT"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 0.0, 1.0)
        b_trial = pm.Normal("b_trial", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)
        tau = pm.HalfNormal("tau", 0.3)

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]

        def _logp(value, mu, sigma, tau):
            x = value - tau
            logp = pm.logp(pm.LogNormal.dist(mu=mu, sigma=sigma), x)
            return pt.switch(x > 0, logp, -np.inf)

        def _random(rng, mu, sigma, tau, size=None):
            mu = np.asarray(mu)
            sigma = np.asarray(sigma)
            tau = np.asarray(tau)
            return rng.lognormal(mean=mu, sigma=sigma, size=size) + tau

        pm.CustomDist(
            "rt_like",
            mu, sigma, tau,
            logp=_logp,
            random=_random,
            observed=RT_obs,
            dims="obs",
        )
    return m


def build_M4_rt_exgaussian(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        RT_obs = pm.Data("RT_obs", d["RT"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 0.0, 1.0)
        b_trial = pm.Normal("b_trial", 0.0, 0.5)

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]
        sigma = pm.HalfNormal("sigma", 0.5)
        nu = pm.HalfNormal("nu", 0.5)

        pm.ExGaussian("rt_like", mu=mu, sigma=sigma, nu=nu, observed=RT_obs, dims="obs")
    return m


# -------------------------
# Joint model builder factory
# -------------------------

def build_joint_model(d, meta, rt_family: str, xN_mode: str):
    """
    rt_family:
      - "normal_logrt"
      - "studentt_logrt"
      - "shifted_lognormal_rt"
      - "exgaussian_rt"
    xN_mode:
      - "logN1p"
      - "N"
    """
    if not np.isfinite(d["N"]).all():
        raise ValueError("Joint models require N.")

    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")

        N_obs = pm.Data("N_obs", d["N"].values.astype(int), dims="obs")

        # shared latent difficulty
        C_image = pm.Normal("C_image", 0.0, 1.0, dims="image")

        # subject RE for N and RT
        subj_re_N, _ = _noncentered_re("subj_re_N", "subject", 0.5)
        subj_re_RT, _ = _noncentered_re("subj_re_RT", "subject", 0.5)

        # N model (anchor sign)
        b0_N = pm.Normal("b0_N", 0.0, 1.0)
        b_trial_N = pm.Normal("b_trial_N", 0.0, 0.5)
        b_C_N = pm.HalfNormal("b_C_N", 0.5)

        eta_N = b0_N + b_trial_N * log_trial + subj_re_N[subj_idx] + b_C_N * C_image[img_idx]
        mu_N = pm.Deterministic("mu_N", pm.math.exp(eta_N), dims="obs")
        alpha_N = pm.HalfNormal("alpha_N", 2.0)
        pm.NegativeBinomial("N_like", mu=mu_N, alpha=alpha_N, observed=N_obs, dims="obs")

        # RT depends on xN
        if xN_mode == "logN1p":
            xN = pm.Data("xN", d["logN1p_c"].values.astype(float), dims="obs")
        elif xN_mode == "N":
            xN = pm.Data("xN", d["N"].values.astype(float), dims="obs")
        else:
            raise ValueError("xN_mode must be 'logN1p' or 'N'")

        b0_RT = pm.Normal("b0_RT", 0.0, 1.0)
        b_trial_RT = pm.Normal("b_trial_RT", 0.0, 0.5)
        b_xN = pm.Normal("b_xN", 0.0, 0.5)
        b_C_RT = pm.Normal("b_C_RT", 0.0, 0.5)

        mu_rt_lin = (
            b0_RT + b_trial_RT * log_trial + subj_re_RT[subj_idx]
            + b_xN * xN + b_C_RT * C_image[img_idx]
        )

        if rt_family == "normal_logrt":
            y = pm.Data("log_rt_obs", d["log_rt"].values, dims="obs")
            sigma_RT = pm.HalfNormal("sigma_RT", 0.5)
            pm.Normal("logrt_like", mu=mu_rt_lin, sigma=sigma_RT, observed=y, dims="obs")

        elif rt_family == "studentt_logrt":
            y = pm.Data("log_rt_obs", d["log_rt"].values, dims="obs")
            sigma_RT = pm.HalfNormal("sigma_RT", 0.5)
            nu = pm.Exponential("nu_minus2_over10", 1 / 10) + 2
            pm.StudentT("logrt_like", nu=nu, mu=mu_rt_lin, sigma=sigma_RT, observed=y, dims="obs")

        elif rt_family == "shifted_lognormal_rt":
            RT_obs = pm.Data("RT_obs", d["RT"].values, dims="obs")
            sigma = pm.HalfNormal("sigma", 0.5)
            tau = pm.HalfNormal("tau", 0.3)

            def _logp(value, mu, sigma, tau):
                x = value - tau
                logp = pm.logp(pm.LogNormal.dist(mu=mu, sigma=sigma), x)
                return pt.switch(x > 0, logp, -np.inf)

            def _random(rng, mu, sigma, tau, size=None):
                mu = np.asarray(mu)
                sigma = np.asarray(sigma)
                tau = np.asarray(tau)
                return rng.lognormal(mean=mu, sigma=sigma, size=size) + tau

            pm.CustomDist(
                "rt_like",
                mu_rt_lin, sigma, tau,
                logp=_logp,
                random=_random,
                observed=RT_obs,
                dims="obs",
            )

        elif rt_family == "exgaussian_rt":
            RT_obs = pm.Data("RT_obs", d["RT"].values, dims="obs")
            sigma = pm.HalfNormal("sigma", 0.5)
            nu = pm.HalfNormal("nu", 0.5)
            pm.ExGaussian("rt_like", mu=mu_rt_lin, sigma=sigma, nu=nu, observed=RT_obs, dims="obs")

        else:
            raise ValueError("Unknown rt_family")

    return m


# -------------------------
# Marginal log-likelihood for Joint Models (for LOO)
# -------------------------

def compute_joint_marginal_loglik(idata, d, meta, rt_family, xN_mode, mcN=20, seed=42):
    """
    Computes log p(RT | params) marginalized over N for joint models.
    Returns an xarray.DataArray compatible with idata.log_likelihood.
    """
    # Extract data
    log_trial = d["log_trial_c"].values
    subj_idx = d["subj_idx"].values
    img_idx = d["img_idx"].values
    y_logrt = d["log_rt"].values.astype(float)
    rt_obs = d["RT"].values.astype(float)
    
    # Prepare centering for N if needed
    ln_mean = 0.0
    if xN_mode == "logN1p" and meta.get("logN1p_mean") is not None:
        ln_mean = float(meta["logN1p_mean"])

    # Extract posterior
    post = idata.posterior
    # Align data to (1, 1, obs) for broadcasting against (chain, draw, 1/obs)
    log_trial = log_trial[None, None, :]
    
    def get_var(name):
        v = post[name].values 
        # Ensure (chain, draw, ...)
        if v.ndim == 2: return v[:, :, None]
        return v

    # RT Linear Predictor components
    b0_RT = get_var("b0_RT")
    b_trial_RT = get_var("b_trial_RT")
    b_xN = get_var("b_xN")
    b_C_RT = get_var("b_C_RT")
    subj_re_RT = post["subj_re_RT"].values[:, :, subj_idx] # (chain, draw, obs)
    C_image = post["C_image"].values[:, :, img_idx]
    
    base_mu = b0_RT + b_trial_RT * log_trial + subj_re_RT + b_C_RT * C_image

    # N parameters
    mu_N = post["mu_N"].values # (chain, draw, obs)
    alpha_N = get_var("alpha_N")

    # RT Scale/Shape parameters
    if rt_family == "normal_logrt":
        sigma_RT = get_var("sigma_RT")
    elif rt_family == "studentt_logrt":
        sigma_RT = get_var("sigma_RT")
        if "nu_minus2_over10" in post:
            nu_RT = post["nu_minus2_over10"].values + 2.0
        elif "nu" in post:
            nu_RT = post["nu"].values
        else:
            nu_RT = 5.0
        if np.ndim(nu_RT) == 2: nu_RT = nu_RT[:, :, None]
    elif rt_family == "shifted_lognormal_rt":
        sigma_RT = get_var("sigma")
        tau_RT = get_var("tau")
    elif rt_family == "exgaussian_rt":
        sigma_RT = get_var("sigma")
        nu_RT = get_var("nu")

    rng = np.random.default_rng(seed)
    
    # Accumulate logsumexp: log( sum exp(logp) )
    # Initialize with -inf
    total_log_prob = np.full(mu_N.shape, -np.inf, dtype=float)

    for _ in range(mcN):
        # Sample N
        lam = rng.gamma(shape=alpha_N, scale=np.clip(mu_N / alpha_N, 1e-12, None))
        N_s = rng.poisson(lam)
        
        if xN_mode == "logN1p":
            xN_s = np.log1p(N_s.astype(float)) - ln_mean
        else:
            xN_s = N_s.astype(float)
            
        mu_rt = base_mu + b_xN * xN_s
        
        # Compute logpdf
        if rt_family == "normal_logrt":
            lp = normal_logpdf(y_logrt[None, None, :], mu_rt, sigma_RT)
        elif rt_family == "studentt_logrt":
            lp = studentt_logpdf(y_logrt[None, None, :], mu_rt, sigma_RT, nu_RT)
        elif rt_family == "shifted_lognormal_rt":
            # Jacobian correction included in helper? No, helper is for RT.
            # We need log p(logRT).
            # shifted_lognormal_logpdf_rt gives log p(RT).
            # log p(logRT) = log p(RT) + logRT
            lp_rt = shifted_lognormal_logpdf_rt(rt_obs[None, None, :], mu_rt, sigma_RT, tau_RT)
            lp = lp_rt + y_logrt[None, None, :]
        elif rt_family == "exgaussian_rt":
            lp_rt = exgaussian_logpdf(rt_obs[None, None, :], mu_rt, sigma_RT, nu_RT)
            lp = lp_rt + y_logrt[None, None, :]
        
        total_log_prob = np.logaddexp(total_log_prob, lp)

    # Average: log(sum) - log(N)
    avg_log_prob = total_log_prob - np.log(mcN)
    
    return xr.DataArray(
        avg_log_prob, 
        dims=["chain", "draw", "obs"], 
        coords={
            "chain": post.coords["chain"],
            "draw": post.coords["draw"],
            "obs": post.coords["obs"]
        }
    )


# -------------------------
# Sampling / loglik helpers
# -------------------------

def fit(model, draws=2000, tune=2000, chains=4, target_accept=0.95, seed=42, progress=True):
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=progress,
            idata_kwargs={"log_likelihood": True},
        )
        if "log_likelihood" not in idata.groups():
            idata = pm.compute_log_likelihood(idata)
    return idata


def ensure_jacobian_logrt_loglik(idata: az.InferenceData, d: pd.DataFrame, rt_ll_var: str, new_name="logrt_like_jac"):
    """
    If rt_ll_var is a pointwise log-likelihood on RT, create equivalent log-likelihood on logRT:
      log p(logRT) = log p(RT) + logRT
    """
    if "log_likelihood" not in idata.groups():
        raise ValueError("InferenceData missing log_likelihood")

    if new_name in idata.log_likelihood:
        return idata

    if rt_ll_var not in idata.log_likelihood:
        raise ValueError(f"Missing {rt_ll_var} in log_likelihood")

    ll = idata.log_likelihood[rt_ll_var]  # dims: chain, draw, obs
    y = xr.DataArray(d["log_rt"].values.astype(float), dims=("obs",))
    ll_new = ll + y  # broadcasts over chain/draw
    idata.log_likelihood[new_name] = ll_new
    return idata


def loo_on_common_logrt(model_name: str, model, idata: az.InferenceData, d: pd.DataFrame):
    """
    Always compute LOO on logRT density.
    If the model likelihood is on RT, add Jacobian correction.
    """
    if "log_likelihood" not in idata.groups():
        with model:
            idata = pm.compute_log_likelihood(idata)

    ll_vars = set(idata.log_likelihood.data_vars)
    
    # Priority: Marginalized RT likelihood (for joint models)
    if "logrt_marginal" in ll_vars:
        return az.loo(idata, var_name="logrt_marginal"), "logrt_marginal"

    if "logrt_like" in ll_vars:
        var = "logrt_like"
        loo = az.loo(idata, var_name=var)
        return loo, var

    # RT-scale likelihood present -> Jacobian correction
    if "rt_like" in ll_vars:
        idata = ensure_jacobian_logrt_loglik(idata, d, "rt_like", new_name="logrt_like_jac")
        var = "logrt_like_jac"
        loo = az.loo(idata, var_name=var)
        return loo, var

    raise ValueError(f"{model_name}: no recognizable RT/logRT likelihood in log_likelihood")


def compare_loo(models: Dict[str, Tuple[pm.Model, az.InferenceData]], d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (m, idata) in models.items():
        loo, var = loo_on_common_logrt(name, m, idata, d)
        rows.append(dict(
            model=name,
            elpd_loo=float(loo.elpd_loo),
            p_loo=float(loo.p_loo),
            se=float(loo.se),
            ll_var=var,
        ))
    out = pd.DataFrame(rows).set_index("model").sort_values("elpd_loo", ascending=False)
    return out


# -------------------------
# CV folds: image-stratified subject×image cell holdout (Option 2)
# -------------------------

@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_image_stratified_cell_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    image_col: str = "img_idx",
    subj_col: str = "subj_idx",
    min_subjects_per_image: int = 2,
) -> List[Fold]:
    """
    For each image, assign its subjects to folds; fold f holds out all rows for (image, subject)
    assigned to f. Ensures each image retains training data in every fold when the image has at
    least min_subjects_per_image unique subjects; otherwise that image is never held out.
    """
    rng = np.random.default_rng(seed)

    img_to_subs = (
        df[[image_col, subj_col]]
        .drop_duplicates()
        .groupby(image_col)[subj_col]
        .apply(lambda x: x.to_numpy())
        .to_dict()
    )

    img_sub_to_fold: Dict[Tuple[int, int], int] = {}
    for img, subs in img_to_subs.items():
        subs = subs.copy()
        rng.shuffle(subs)

        if len(subs) < min_subjects_per_image:
            continue

        for j, s in enumerate(subs):
            img_sub_to_fold[(int(img), int(s))] = j % n_splits

    all_idx = np.arange(len(df))
    img_vals = df[image_col].to_numpy().astype(int)
    subj_vals = df[subj_col].to_numpy().astype(int)

    folds: List[Fold] = []
    for f in range(n_splits):
        is_test = np.zeros(len(df), dtype=bool)
        for t in range(len(df)):
            key = (img_vals[t], subj_vals[t])
            if key in img_sub_to_fold and img_sub_to_fold[key] == f:
                is_test[t] = True

        test_idx = all_idx[is_test]
        train_idx = all_idx[~is_test]
        folds.append(Fold(fold_id=f, train_idx=train_idx, test_idx=test_idx))
    return folds


# -------------------------
# Ranking stability (uncertainty-aware)
# -------------------------

def extract_image_effect_samples(idata: az.InferenceData, var_name: str) -> np.ndarray:
    """
    Returns samples (S, I) where S=chain*draw.
    """
    x = idata.posterior[var_name].stack(sample=("chain", "draw")).values
    # x is either (I,S) or (S,I)
    if x.shape[0] == idata.posterior[var_name].sizes.get("image", x.shape[0]):
        x = np.moveaxis(x, 0, 1)  # (S,I)
    return x


def pairwise_order_prob(samples: np.ndarray) -> np.ndarray:
    """
    samples: (S,I) -> P(I,I) with P[i,j]=Pr(i>j)
    """
    S, I = samples.shape
    P = (samples[:, :, None] > samples[:, None, :]).mean(axis=0)
    np.fill_diagonal(P, 0.5)
    return P


def ranking_stability_from_fold_samples(fold_samples: List[np.ndarray], topk: int = 20, ci: float = 0.95) -> Dict[str, float]:
    """
    Computes:
      - mean Spearman over folds (using ranks of posterior means)
      - Kendall-like tau over folds
      - top-k overlap
      - stability of pairwise probabilities (mean abs diff)
      - CI overlap fraction
    """
    F = len(fold_samples)
    if F < 2:
        return {"n_folds": F}

    means = [fs.mean(axis=0) for fs in fold_samples]
    ranks = [np.argsort(np.argsort(-m)) for m in means]  # 0=hardest

    # CIs per fold
    alpha = (1 - ci) / 2
    cis = [(np.quantile(fs, alpha, axis=0), np.quantile(fs, 1 - alpha, axis=0)) for fs in fold_samples]
    Ps = [pairwise_order_prob(fs) for fs in fold_samples]

    spears, ktaus, top_over, prob_d, ci_over = [], [], [], [], []

    for a in range(F):
        for b in range(a + 1, F):
            ra, rb = ranks[a], ranks[b]
            spears.append(float(np.corrcoef(ra, rb)[0, 1]))

            # Kendall-like tau (pairwise concordance)
            I = len(ra)
            concord = 0
            discord = 0
            for i in range(I):
                for j in range(i + 1, I):
                    sa = np.sign(ra[i] - ra[j])
                    sb = np.sign(rb[i] - rb[j])
                    if sa == sb:
                        concord += 1
                    else:
                        discord += 1
            denom = concord + discord
            ktaus.append(float((concord - discord) / denom) if denom else np.nan)

            ka = set(np.argsort(-means[a])[: min(topk, I)])
            kb = set(np.argsort(-means[b])[: min(topk, I)])
            top_over.append(len(ka & kb) / max(1, min(topk, I)))

            tri = np.triu_indices(I, k=1)
            prob_d.append(float(np.mean(np.abs(Ps[a][tri] - Ps[b][tri]))))

            loa, hia = cis[a]
            lob, hib = cis[b]
            ci_over.append(float(np.mean((loa <= hib) & (lob <= hia))))

    return dict(
        n_folds=int(F),
        spearman_rank_mean=float(np.nanmean(spears)),
        kendall_tau_mean=float(np.nanmean(ktaus)),
        topk_overlap_mean=float(np.nanmean(top_over)),
        pairprob_absdiff_mean=float(np.nanmean(prob_d)),
        ci_overlap_frac_mean=float(np.nanmean(ci_over)),
    )


# -------------------------
# CV scoring (marginal over N for joint models)
# -------------------------

def _stack_post(idata: az.InferenceData, var: str) -> np.ndarray:
    return idata.posterior[var].stack(sample=("chain", "draw")).values


def _ensure_S_first(a: np.ndarray, S: int) -> np.ndarray:
    # if last dim is S, move it
    if a.shape[0] != S and a.shape[-1] == S:
        return np.moveaxis(a, -1, 0)
    return a


def score_logrt_mixture_normal(y, mu_SxT, sigma_S):
    """
    y: (T,)
    mu_SxT: (S,T)
    sigma_S: (S,)
    returns: rmse, elpd_total
    """
    y = y[None, :]
    sigma = sigma_S[:, None]
    logpdf = normal_logpdf(y, mu_SxT, sigma)
    lpd = logsumexp(logpdf, axis=0) - np.log(logpdf.shape[0])
    elpd = float(np.sum(lpd))
    yhat = mu_SxT.mean(axis=0)
    rmse = float(np.sqrt(np.mean((y.squeeze(0) - yhat) ** 2)))
    return rmse, elpd


def score_logrt_mixture_studentt(y, mu_SxT, sigma_S, nu_S):
    y = y[None, :]
    sigma = sigma_S[:, None]
    nu = nu_S[:, None]
    logpdf = studentt_logpdf(y, mu_SxT, sigma, nu)
    lpd = logsumexp(logpdf, axis=0) - np.log(logpdf.shape[0])
    elpd = float(np.sum(lpd))
    yhat = mu_SxT.mean(axis=0)
    rmse = float(np.sqrt(np.mean((y.squeeze(0) - yhat) ** 2)))
    return rmse, elpd


def score_logrt_mixture_shifted_lognormal(y_logrt, rt, mu_SxT, sigma_S, tau_S):
    """
    Evaluate log p(logRT) = log p(RT) + logRT, with RT = exp(logRT).
    """
    rt = rt[None, :]
    mu = mu_SxT
    sigma = sigma_S[:, None]
    tau = tau_S[:, None]
    logp_rt = shifted_lognormal_logpdf_rt(rt, mu, sigma, tau)
    logp_logrt = logp_rt + y_logrt[None, :]
    lpd = logsumexp(logp_logrt, axis=0) - np.log(logp_logrt.shape[0])
    elpd = float(np.sum(lpd))

    # predictive mean for logRT: approximate via mean RT then log
    rt_mean = (tau + np.exp(mu + 0.5 * sigma * sigma)).mean(axis=0)
    yhat = np.log(np.clip(rt_mean, 1e-12, None))
    rmse = float(np.sqrt(np.mean((y_logrt - yhat) ** 2)))
    return rmse, elpd


def score_logrt_mixture_exgaussian(y_logrt, rt, mu_SxT, sigma_S, nu_S):
    rt = rt[None, :]
    sigma = sigma_S[:, None]
    nu = nu_S[:, None]
    logp_rt = exgaussian_logpdf(rt, mu_SxT, sigma, nu)
    logp_logrt = logp_rt + y_logrt[None, :]
    lpd = logsumexp(logp_logrt, axis=0) - np.log(logp_logrt.shape[0])
    elpd = float(np.sum(lpd))

    # mean of ExGaussian is mu + nu
    rt_mean = (mu_SxT + nu).mean(axis=0)
    yhat = np.log(np.clip(rt_mean, 1e-12, None))
    rmse = float(np.sqrt(np.mean((y_logrt - yhat) ** 2)))
    return rmse, elpd


def cellheldout_cv(
    d: pd.DataFrame,
    meta: dict,
    model_builders: Dict[str, Tuple],
    n_splits: int = 5,
    seed: int = 42,
    draws: int = 600,
    tune: int = 600,
    chains: int = 2,
    target_accept: float = 0.95,
    mcN: int = 2,
    topk: int = 20,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    """
    Runs image-stratified cell-held-out CV for each model.
    model_builders: dict[name] = (builder_fn, builder_kwargs, model_kind, img_effect_var, rt_family, xN_mode)
      model_kind in {"rt_only", "joint"}
      rt_family in {"normal_logrt","studentt_logrt","shifted_lognormal_rt","exgaussian_rt"}
      xN_mode only for joint.
    Returns per-model summary including ranking stability computed from fold idatas.
    """
    from complexity.models.sampling import sample_fold

    folds = make_image_stratified_cell_folds(d, n_splits=n_splits, seed=seed)

    rng = np.random.default_rng(seed)
    results: Dict[str, dict] = {}

    for name, (builder, kwargs, model_kind, img_var, rt_family, xN_mode) in model_builders.items():
        rmse_f, elpd_f = [], []
        fold_img_samples = []

        for fold in folds:
            d_tr = d.iloc[fold.train_idx].copy()
            d_te = d.iloc[fold.test_idx].copy()
            if len(d_te) == 0:
                continue

            # fold centering
            lt_mean = float(d_tr["log_trial"].mean())
            d_tr["log_trial_c"] = d_tr["log_trial"] - lt_mean
            d_te["log_trial_c"] = d_te["log_trial"] - lt_mean

            ln_mean = None
            if meta["has_N"]:
                ln_mean = float(d_tr["logN1p"].mean())
                d_tr["logN1p_c"] = d_tr["logN1p"] - ln_mean
                d_te["logN1p_c"] = d_te["logN1p"] - ln_mean

            # build + fit
            m = builder(d_tr, meta, **kwargs)
            with m:
                idata = sample_fold(
                    m, d_tr, d_te,
                    checkpoint_dir / name / f'fold_{fold.fold_id}' if checkpoint_dir is not None else None,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=int(rng.integers(1, 1_000_000)),
                    return_inferencedata=True,
                    progressbar=True,
                )

            # extract image effect samples for ranking stability
            try:
                fold_img_samples.append(extract_image_effect_samples(idata, img_var))
            except Exception:
                pass

            # score held-out
            y = d_te["log_rt"].values.astype(float)
            rt = d_te["RT"].values.astype(float)
            subj = d_te["subj_idx"].values.astype(int)
            img = d_te["img_idx"].values.astype(int)
            log_trial = d_te["log_trial_c"].values.astype(float)

            post = idata.posterior.stack(sample=("chain", "draw"))
            S = post.sizes["sample"]

            if model_kind == "rt_only":
                # mu for each posterior draw
                b0 = post["b0"].values.astype(float)  # (S,)
                b_trial = post["b_trial"].values.astype(float)  # (S,)
                subj_re = _ensure_S_first(post["subj_re"].values.astype(float), S)  # (S, nsubj)
                img_re = _ensure_S_first(post["img_re"].values.astype(float), S)    # (S, nimg)
                mu = (
                    b0[:, None]
                    + b_trial[:, None] * log_trial[None, :]
                    + subj_re[:, subj]
                    + img_re[:, img]
                )

                if rt_family == "normal_logrt":
                    sigma = post["sigma"].values.astype(float)
                    rmse, elpd = score_logrt_mixture_normal(y, mu, sigma)

                elif rt_family == "studentt_logrt":
                    sigma = post["sigma"].values.astype(float)
                    nu = post["nu_minus2_over10"].values.astype(float) * 0.0  # placeholder
                    # model stores nu as deterministic expression; in idata it is 'nu_minus2_over10'? no.
                    # safer: recompute nu from stored variable if present
                    if "nu_minus2_over10" in post:
                        nu = post["nu_minus2_over10"].values.astype(float) + 2.0
                    elif "nu" in post:
                        nu = post["nu"].values.astype(float)
                    else:
                        # if missing, approximate dof=5
                        nu = np.full(S, 5.0)
                    rmse, elpd = score_logrt_mixture_studentt(y, mu, sigma, nu)

                elif rt_family == "shifted_lognormal_rt":
                    sigma = post["sigma"].values.astype(float)
                    tau = post["tau"].values.astype(float)
                    rmse, elpd = score_logrt_mixture_shifted_lognormal(y, rt, mu, sigma, tau)

                elif rt_family == "exgaussian_rt":
                    sigma = post["sigma"].values.astype(float)
                    nu = post["nu"].values.astype(float)
                    rmse, elpd = score_logrt_mixture_exgaussian(y, rt, mu, sigma, nu)

                else:
                    raise ValueError("Unknown rt_family")

            else:
                # joint: marginalize N
                b0_RT = post["b0_RT"].values.astype(float)
                b_trial_RT = post["b_trial_RT"].values.astype(float)
                b_xN = post["b_xN"].values.astype(float)
                b_C_RT = post["b_C_RT"].values.astype(float)

                subj_re_RT = _ensure_S_first(post["subj_re_RT"].values.astype(float), S)
                C_image = _ensure_S_first(post["C_image"].values.astype(float), S)

                # Recompute mu_N on held-out using posterior parameters (avoid dependence on training obs length):
                b0_N = post["b0_N"].values.astype(float)
                b_trial_N = post["b_trial_N"].values.astype(float)
                b_C_N = post["b_C_N"].values.astype(float)
                subj_re_N = _ensure_S_first(post["subj_re_N"].values.astype(float), S)
                alpha_N = post["alpha_N"].values.astype(float)

                etaN_te = (
                    b0_N[:, None]
                    + b_trial_N[:, None] * log_trial[None, :]
                    + subj_re_N[:, subj]
                    + b_C_N[:, None] * C_image[:, img]
                )
                muN_te = np.exp(etaN_te)

                # base mu for RT (excluding xN)
                base = (
                    b0_RT[:, None]
                    + b_trial_RT[:, None] * log_trial[None, :]
                    + subj_re_RT[:, subj]
                    + b_C_RT[:, None] * C_image[:, img]
                )

                # MC over N per posterior draw
                # combine posterior draws and mcN into one dimension for mixture evaluation
                mu_all = []
                if xN_mode == "logN1p":
                    for _ in range(mcN):
                        N_s = negbin_gamma_poisson_sample(muN_te, alpha_N[:, None], rng)
                        xN = np.log1p(N_s.astype(float)) - float(ln_mean if ln_mean is not None else 0.0)
                        mu_all.append(base + b_xN[:, None] * xN)
                elif xN_mode == "N":
                    for _ in range(mcN):
                        N_s = negbin_gamma_poisson_sample(muN_te, alpha_N[:, None], rng)
                        mu_all.append(base + b_xN[:, None] * N_s.astype(float))
                else:
                    raise ValueError("bad xN_mode")

                # Each MC block contains every posterior draw in its original order.
                mu_mc = np.concatenate(mu_all, axis=0)  # (S*mcN, T)
                if rt_family == "normal_logrt":
                    sigma = post["sigma_RT"].values.astype(float)
                    sigma_mc = np.tile(sigma, mcN)
                    rmse, elpd = score_logrt_mixture_normal(y, mu_mc, sigma_mc)

                elif rt_family == "studentt_logrt":
                    sigma = post["sigma_RT"].values.astype(float)
                    sigma_mc = np.tile(sigma, mcN)
                    if "nu_minus2_over10" in post:
                        nu = post["nu_minus2_over10"].values.astype(float) + 2.0
                    elif "nu" in post:
                        nu = post["nu"].values.astype(float)
                    else:
                        nu = np.full(S, 5.0)
                    nu_mc = np.tile(nu, mcN)
                    rmse, elpd = score_logrt_mixture_studentt(y, mu_mc, sigma_mc, nu_mc)

                elif rt_family == "shifted_lognormal_rt":
                    sigma = post["sigma"].values.astype(float)
                    tau = post["tau"].values.astype(float)
                    sigma_mc = np.tile(sigma, mcN)
                    tau_mc = np.tile(tau, mcN)
                    rmse, elpd = score_logrt_mixture_shifted_lognormal(y, rt, mu_mc, sigma_mc, tau_mc)

                elif rt_family == "exgaussian_rt":
                    sigma = post["sigma"].values.astype(float)
                    nu = post["nu"].values.astype(float)
                    sigma_mc = np.tile(sigma, mcN)
                    nu_mc = np.tile(nu, mcN)
                    rmse, elpd = score_logrt_mixture_exgaussian(y, rt, mu_mc, sigma_mc, nu_mc)

                else:
                    raise ValueError("Unknown rt_family")

            rmse_f.append(rmse)
            elpd_f.append(elpd)

        stability = ranking_stability_from_fold_samples(fold_img_samples, topk=topk)

        results[name] = dict(
            rmse_logrt_mean=float(np.mean(rmse_f)) if rmse_f else np.nan,
            elpd_logrt_total_mean=float(np.mean(elpd_f)) if elpd_f else np.nan,
            n_folds_scored=int(len(rmse_f)),
            ranking_stability=stability,
        )

    return results


# -------------------------
# Automatic model selection table
# -------------------------

def build_model_selection_table(cv_results: Dict[str, dict]) -> pd.DataFrame:
    """
    Combines accuracy + stability into one table and picks best by composite rank score.
    Uses rank-based (robust) normalization.

    Composite score rewards:
      - higher spearman_rank_mean
      - lower rmse_logrt_mean
      - lower pairprob_absdiff_mean
      - higher ci_overlap_frac_mean
    """
    rows = []
    for model, res in cv_results.items():
        st = res.get("ranking_stability", {})
        rows.append(dict(
            model=model,
            rmse_logrt_mean=res.get("rmse_logrt_mean", np.nan),
            elpd_logrt_total_mean=res.get("elpd_logrt_total_mean", np.nan),
            n_folds_scored=res.get("n_folds_scored", 0),
            spearman_rank_mean=st.get("spearman_rank_mean", np.nan),
            kendall_tau_mean=st.get("kendall_tau_mean", np.nan),
            topk_overlap_mean=st.get("topk_overlap_mean", np.nan),
            pairprob_absdiff_mean=st.get("pairprob_absdiff_mean", np.nan),
            ci_overlap_frac_mean=st.get("ci_overlap_frac_mean", np.nan),
        ))
    df = pd.DataFrame(rows)

    # rank helpers (0..1 where 1 is best)
    def rank01(s, higher_better=True):
        s = s.copy()
        # handle NaNs by putting them worst
        nan_mask = ~np.isfinite(s.values)
        if nan_mask.any():
            fill = np.nanmin(s.values[np.isfinite(s.values)]) if higher_better else np.nanmax(s.values[np.isfinite(s.values)])
            s.values[nan_mask] = fill - 1e9 if higher_better else fill + 1e9

        r = s.rank(ascending=not higher_better, method="average")
        # map to [0,1], 1 best
        return 1.0 - (r - 1.0) / max(1.0, (len(r) - 1.0))

    df["rank_spearman"] = rank01(df["spearman_rank_mean"], higher_better=True)
    df["rank_rmse"] = rank01(df["rmse_logrt_mean"], higher_better=False)
    df["rank_pairprob"] = rank01(df["pairprob_absdiff_mean"], higher_better=False)
    df["rank_ci"] = rank01(df["ci_overlap_frac_mean"], higher_better=True)

    # composite (weights emphasize complexity ranking stability while keeping accuracy)
    df["composite_score"] = (
        0.40 * df["rank_spearman"]
        + 0.30 * df["rank_rmse"]
        + 0.20 * df["rank_pairprob"]
        + 0.10 * df["rank_ci"]
    )

    # Pareto flag on (spearman high, rmse low)
    df["pareto"] = False
    vals = df[["spearman_rank_mean", "rmse_logrt_mean"]].to_numpy()
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            # j dominates i if spearman_j >= spearman_i and rmse_j <= rmse_i with at least one strict
            if (vals[j, 0] >= vals[i, 0]) and (vals[j, 1] <= vals[i, 1]) and ((vals[j, 0] > vals[i, 0]) or (vals[j, 1] < vals[i, 1])):
                dominated = True
                break
        df.loc[i, "pareto"] = not dominated

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["best_by_composite"] = False
    if len(df):
        df.loc[0, "best_by_composite"] = True
    return df


# -------------------------
# Main
# -------------------------

def main() -> None:
  """Run the preserved model definitions through the configurable suite runner."""
  import sys

  if __package__:
    from .runner import run_suite
  else:
    from runner import run_suite

  run_suite(sys.modules[__name__], 'rt')


if __name__ == '__main__':
  main()
