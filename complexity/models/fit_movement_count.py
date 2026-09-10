#!/usr/bin/env python3
# ============================================================
# Count-as-Continuous model suite for COCO-style visual search data
# Predicting Fixation Counts (N) using RT-style continuous models
# ============================================================
# Models (Joint models removed):
#     M1  log(N) ~ Normal
#     M2  log(N) ~ StudentT
#     M3  N ~ ShiftedLogNormal (N = tau + LogNormal)
#     M4  N ~ ExGaussian
#
# Comparison:
#   - PSIS-LOO on a common target: log(N) density.
#     For models whose likelihood is specified on N (M3/M4), we apply
#     Jacobian correction: log p(logN) = log p(N) + logN.
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
import pytensor.tensor as pt
import math

# -------------------------
# Numeric utilities
# -------------------------

try:
    from scipy.special import logsumexp as _logsumexp
except Exception:
    _logsumexp = None

def logsumexp(a, axis=None):
    if _logsumexp is not None:
        return _logsumexp(a, axis=axis)
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def log_ndtr(z):
    return np.log(0.5 * np.clip(np.vectorize(math.erfc)(-z / np.sqrt(2.0)), 1e-300, 1.0))

def exgaussian_logpdf(x, mu, sigma, nu):
    x, mu, sigma, nu = np.asarray(x), np.asarray(mu), np.asarray(sigma), np.asarray(nu)
    z = (x - mu) / sigma - sigma / nu
    logf = -np.log(nu) + (sigma * sigma) / (2.0 * nu * nu) - (x - mu) / nu + log_ndtr(z)
    return logf

def studentt_logpdf(y, mu, sigma, nu):
    y, mu, sigma, nu = np.asarray(y), np.asarray(mu), np.asarray(sigma), np.asarray(nu)
    lgamma = np.vectorize(math.lgamma)
    a = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)
    b = -0.5 * np.log(nu * np.pi) - np.log(sigma)
    z2 = ((y - mu) / sigma) ** 2
    c = -0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    return a + b + c

def normal_logpdf(y, mu, sigma):
    y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2

def shifted_lognormal_logpdf_n(n_val, mu, sigma, tau):
    n_val = np.asarray(n_val)
    x = n_val - tau
    out = np.full_like(x, -np.inf, dtype=float)
    ok = x > 0
    mu_bc = np.broadcast_to(mu, x.shape)
    sigma_bc = np.broadcast_to(sigma, x.shape)
    
    x_val = x[ok]
    mu_val = mu_bc[ok]
    sigma_val = sigma_bc[ok]
    lx = np.log(x_val)
    
    out[ok] = (
        -lx - np.log(sigma_val) - 0.5 * np.log(2 * np.pi) - 0.5 * ((lx - mu_val) / sigma_val) ** 2
    )
    return out

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
    n_col="N",
    subj_col="subject",
    img_col="image",
    trial_col="trial",
):
    d = df.copy()

    # Required columns
    req = [n_col, subj_col, img_col, trial_col]
    d = d.dropna(subset=req).copy()
    
    # Process N: treat as continuous, must be > 0 for logs
    d[n_col] = pd.to_numeric(d[n_col], errors="coerce")
    d = d.dropna(subset=[n_col]).copy()
    d = d[d[n_col] > 0].copy() # Filter N>0 to avoid log(0)
    d["N"] = d[n_col].astype(float)
    d["log_n"] = np.log(d["N"].values)

    d[subj_col] = d[subj_col].astype(str)
    d[img_col] = d[img_col].astype(str)
    
    subj_idx, subj_levels = pd.factorize(d[subj_col], sort=True)
    img_idx, img_levels = pd.factorize(d[img_col], sort=True)
    d["subj_idx"] = subj_idx.astype(int)
    d["img_idx"] = img_idx.astype(int)

    d["trial"] = pd.to_numeric(d[trial_col], errors="coerce")
    d = d.dropna(subset=["trial"]).copy()
    d["log_trial"] = np.log1p(d["trial"].astype(float).values)

    lt_mean = float(d["log_trial"].mean())
    d["log_trial_c"] = d["log_trial"] - lt_mean

    meta = dict(
        subjects=subj_levels.tolist(),
        images=img_levels.tolist(),
        log_trial_mean=lt_mean,
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
# Model builders (N-continuous)
# -------------------------

def build_M1_n_lognormal(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        y = pm.Data("log_n_obs", d["log_n"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 2.0, 1.0) # Adjusted prior for log(N) ~ 2.0
        b_trial = pm.Normal("b_trial", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]
        pm.Normal("logn_like", mu=mu, sigma=sigma, observed=y, dims="obs")
    return m

def build_M2_n_studentt(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        y = pm.Data("log_n_obs", d["log_n"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 2.0, 1.0)
        b_trial = pm.Normal("b_trial", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)
        nu = pm.Exponential("nu_minus2_over10", 1 / 10) + 2

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]
        pm.StudentT("logn_like", nu=nu, mu=mu, sigma=sigma, observed=y, dims="obs")
    return m

def build_M3_n_shifted_lognormal(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        N_obs = pm.Data("N_obs", d["N"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 2.0, 1.0)
        b_trial = pm.Normal("b_trial", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)
        tau = pm.HalfNormal("tau", 0.5) # Shift parameter for N

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]

        def _logp(value, mu, sigma, tau):
            x = value - tau
            logp = pm.logp(pm.LogNormal.dist(mu=mu, sigma=sigma), x)
            return pt.switch(x > 0, logp, -np.inf)

        def _random(rng, mu, sigma, tau, size=None):
            return rng.lognormal(mean=mu, sigma=sigma, size=size) + tau

        pm.CustomDist(
            "n_like",
            mu, sigma, tau,
            logp=_logp,
            random=_random,
            observed=N_obs,
            dims="obs",
        )
    return m

def build_M4_n_exgaussian(d, meta):
    coords = {"subject": meta["subjects"], "image": meta["images"], "obs": np.arange(len(d))}
    with pm.Model(coords=coords) as m:
        subj_idx = pm.Data("subj_idx", d["subj_idx"].values, dims="obs")
        img_idx = pm.Data("img_idx", d["img_idx"].values, dims="obs")
        log_trial = pm.Data("log_trial", d["log_trial_c"].values, dims="obs")
        N_obs = pm.Data("N_obs", d["N"].values, dims="obs")

        subj_re, _ = _noncentered_re("subj_re", "subject", 0.5)
        img_re, _ = _noncentered_re("img_re", "image", 0.5)

        b0 = pm.Normal("b0", 10.0, 5.0) # Linear scale for ExGauss mu component
        b_trial = pm.Normal("b_trial", 0.0, 1.0)

        mu = b0 + b_trial * log_trial + subj_re[subj_idx] + img_re[img_idx]
        sigma = pm.HalfNormal("sigma", 5.0)
        nu = pm.HalfNormal("nu", 5.0)

        pm.ExGaussian("n_like", mu=mu, sigma=sigma, nu=nu, observed=N_obs, dims="obs")
    return m

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

def ensure_jacobian_logn_loglik(idata: az.InferenceData, d: pd.DataFrame, n_ll_var: str, new_name="logn_like_jac"):
    """
    If n_ll_var is a pointwise log-likelihood on N, create equivalent log-likelihood on logN:
      log p(logN) = log p(N) + logN
    """
    if "log_likelihood" not in idata.groups():
        raise ValueError("InferenceData missing log_likelihood")

    if new_name in idata.log_likelihood:
        return idata

    ll = idata.log_likelihood[n_ll_var]
    y = xr.DataArray(d["log_n"].values.astype(float), dims=("obs",))
    ll_new = ll + y 
    idata.log_likelihood[new_name] = ll_new
    return idata

def loo_on_common_logn(model_name: str, model, idata: az.InferenceData, d: pd.DataFrame):
    """
    Compute LOO on logN density.
    """
    if "log_likelihood" not in idata.groups():
        with model:
            idata = pm.compute_log_likelihood(idata)

    ll_vars = set(idata.log_likelihood.data_vars)
    
    if "logn_like" in ll_vars:
        return az.loo(idata, var_name="logn_like"), "logn_like"

    if "n_like" in ll_vars:
        idata = ensure_jacobian_logn_loglik(idata, d, "n_like", new_name="logn_like_jac")
        return az.loo(idata, var_name="logn_like_jac"), "logn_like_jac"

    raise ValueError(f"{model_name}: no recognizable N/logN likelihood")

def compare_loo(models: Dict[str, Tuple[pm.Model, az.InferenceData]], d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (m, idata) in models.items():
        loo, var = loo_on_common_logn(name, m, idata, d)
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
# CV folds
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
    rng = np.random.default_rng(seed)
    img_to_subs = (
        df[[image_col, subj_col]]
        .drop_duplicates()
        .groupby(image_col)[subj_col]
        .apply(lambda x: x.to_numpy())
        .to_dict()
    )

    img_sub_to_fold = {}
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

    folds = []
    for f in range(n_splits):
        is_test = np.zeros(len(df), dtype=bool)
        for t in range(len(df)):
            key = (img_vals[t], subj_vals[t])
            if key in img_sub_to_fold and img_sub_to_fold[key] == f:
                is_test[t] = True
        folds.append(Fold(f, all_idx[~is_test], all_idx[is_test]))
    return folds

# -------------------------
# Ranking stability
# -------------------------

def extract_image_effect_samples(idata: az.InferenceData, var_name: str) -> np.ndarray:
    x = idata.posterior[var_name].stack(sample=("chain", "draw")).values
    if x.shape[0] == idata.posterior[var_name].sizes.get("image", x.shape[0]):
        x = np.moveaxis(x, 0, 1)  # (S,I)
    return x

def pairwise_order_prob(samples: np.ndarray) -> np.ndarray:
    S, I = samples.shape
    P = (samples[:, :, None] > samples[:, None, :]).mean(axis=0)
    np.fill_diagonal(P, 0.5)
    return P

def ranking_stability_from_fold_samples(fold_samples: List[np.ndarray], topk: int = 20, ci: float = 0.95) -> Dict[str, float]:
    F = len(fold_samples)
    if F < 2:
        return {"n_folds": F}

    means = [fs.mean(axis=0) for fs in fold_samples]
    ranks = [np.argsort(np.argsort(-m)) for m in means] 

    alpha = (1 - ci) / 2
    cis = [(np.quantile(fs, alpha, axis=0), np.quantile(fs, 1 - alpha, axis=0)) for fs in fold_samples]
    Ps = [pairwise_order_prob(fs) for fs in fold_samples]

    spears, ktaus, top_over, prob_d, ci_over = [], [], [], [], []

    for a in range(F):
        for b in range(a + 1, F):
            ra, rb = ranks[a], ranks[b]
            spears.append(float(np.corrcoef(ra, rb)[0, 1]))

            I = len(ra)
            concord, discord = 0, 0
            for i in range(I):
                for j in range(i + 1, I):
                    if np.sign(ra[i] - ra[j]) == np.sign(rb[i] - rb[j]):
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
# CV scoring (N)
# -------------------------

def _ensure_S_first(a: np.ndarray, S: int) -> np.ndarray:
    if a.shape[0] != S and a.shape[-1] == S:
        return np.moveaxis(a, -1, 0)
    return a

def score_logn_mixture_normal(y, mu_SxT, sigma_S):
    y = y[None, :]
    sigma = sigma_S[:, None]
    logpdf = normal_logpdf(y, mu_SxT, sigma)
    lpd = logsumexp(logpdf, axis=0) - np.log(logpdf.shape[0])
    elpd = float(np.sum(lpd))
    yhat = mu_SxT.mean(axis=0)
    rmse = float(np.sqrt(np.mean((y.squeeze(0) - yhat) ** 2)))
    return rmse, elpd

def score_logn_mixture_studentt(y, mu_SxT, sigma_S, nu_S):
    y = y[None, :]
    sigma = sigma_S[:, None]
    nu = nu_S[:, None]
    logpdf = studentt_logpdf(y, mu_SxT, sigma, nu)
    lpd = logsumexp(logpdf, axis=0) - np.log(logpdf.shape[0])
    elpd = float(np.sum(lpd))
    yhat = mu_SxT.mean(axis=0)
    rmse = float(np.sqrt(np.mean((y.squeeze(0) - yhat) ** 2)))
    return rmse, elpd

def score_logn_mixture_shifted_lognormal(y_logn, n_val, mu_SxT, sigma_S, tau_S):
    n_val = n_val[None, :]
    mu = mu_SxT
    sigma = sigma_S[:, None]
    tau = tau_S[:, None]
    logp_n = shifted_lognormal_logpdf_n(n_val, mu, sigma, tau)
    logp_logn = logp_n + y_logn[None, :]
    lpd = logsumexp(logp_logn, axis=0) - np.log(logp_logn.shape[0])
    elpd = float(np.sum(lpd))
    n_mean = (tau + np.exp(mu + 0.5 * sigma * sigma)).mean(axis=0)
    yhat = np.log(np.clip(n_mean, 1e-12, None))
    rmse = float(np.sqrt(np.mean((y_logn - yhat) ** 2)))
    return rmse, elpd

def score_logn_mixture_exgaussian(y_logn, n_val, mu_SxT, sigma_S, nu_S):
    n_val = n_val[None, :]
    sigma = sigma_S[:, None]
    nu = nu_S[:, None]
    logp_n = exgaussian_logpdf(n_val, mu_SxT, sigma, nu)
    logp_logn = logp_n + y_logn[None, :]
    lpd = logsumexp(logp_logn, axis=0) - np.log(logp_logn.shape[0])
    elpd = float(np.sum(lpd))
    n_mean = (mu_SxT + nu).mean(axis=0)
    yhat = np.log(np.clip(n_mean, 1e-12, None))
    rmse = float(np.sqrt(np.mean((y_logn - yhat) ** 2)))
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
    topk: int = 20,
) -> Dict[str, dict]:
    folds = make_image_stratified_cell_folds(d, n_splits=n_splits, seed=seed)
    rng = np.random.default_rng(seed)
    results = {}

    for name, (builder, kwargs, family, img_var) in model_builders.items():
        rmse_f, elpd_f = [], []
        fold_img_samples = []

        for fold in folds:
            d_tr = d.iloc[fold.train_idx].copy()
            d_te = d.iloc[fold.test_idx].copy()
            if len(d_te) == 0:
                continue

            lt_mean = float(d_tr["log_trial"].mean())
            d_tr["log_trial_c"] = d_tr["log_trial"] - lt_mean
            d_te["log_trial_c"] = d_te["log_trial"] - lt_mean

            m = builder(d_tr, meta, **kwargs)
            with m:
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=int(rng.integers(1, 1_000_000)),
                    return_inferencedata=True,
                    progressbar=True,
                )

            try:
                fold_img_samples.append(extract_image_effect_samples(idata, img_var))
            except Exception:
                pass

            y = d_te["log_n"].values.astype(float)
            n_val = d_te["N"].values.astype(float)
            subj = d_te["subj_idx"].values.astype(int)
            img = d_te["img_idx"].values.astype(int)
            log_trial = d_te["log_trial_c"].values.astype(float)

            post = idata.posterior.stack(sample=("chain", "draw"))
            S = post.sizes["sample"]

            b0 = post["b0"].values.astype(float)
            b_trial = post["b_trial"].values.astype(float)
            subj_re = _ensure_S_first(post["subj_re"].values.astype(float), S)
            img_re = _ensure_S_first(post["img_re"].values.astype(float), S)

            mu = (
                b0[:, None]
                + b_trial[:, None] * log_trial[None, :]
                + subj_re[:, subj]
                + img_re[:, img]
            )

            if family == "normal_logn":
                sigma = post["sigma"].values.astype(float)
                rmse, elpd = score_logn_mixture_normal(y, mu, sigma)

            elif family == "studentt_logn":
                sigma = post["sigma"].values.astype(float)
                if "nu_minus2_over10" in post:
                    nu = post["nu_minus2_over10"].values.astype(float) + 2.0
                elif "nu" in post:
                    nu = post["nu"].values.astype(float)
                else:
                    nu = np.full(S, 5.0)
                rmse, elpd = score_logn_mixture_studentt(y, mu, sigma, nu)

            elif family == "shifted_lognormal_n":
                sigma = post["sigma"].values.astype(float)
                tau = post["tau"].values.astype(float)
                rmse, elpd = score_logn_mixture_shifted_lognormal(y, n_val, mu, sigma, tau)

            elif family == "exgaussian_n":
                sigma = post["sigma"].values.astype(float)
                nu = post["nu"].values.astype(float)
                rmse, elpd = score_logn_mixture_exgaussian(y, n_val, mu, sigma, nu)

            else:
                raise ValueError("Unknown family")

            rmse_f.append(rmse)
            elpd_f.append(elpd)

        stability = ranking_stability_from_fold_samples(fold_img_samples, topk=topk)
        results[name] = dict(
            rmse_logn_mean=float(np.mean(rmse_f)) if rmse_f else np.nan,
            elpd_logn_total_mean=float(np.mean(elpd_f)) if elpd_f else np.nan,
            n_folds_scored=int(len(rmse_f)),
            ranking_stability=stability,
        )

    return results

# -------------------------
# Selection table
# -------------------------

def build_model_selection_table(cv_results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, res in cv_results.items():
        st = res.get("ranking_stability", {})
        rows.append(dict(
            model=model,
            rmse_logn_mean=res.get("rmse_logn_mean", np.nan),
            elpd_logn_total_mean=res.get("elpd_logn_total_mean", np.nan),
            n_folds_scored=res.get("n_folds_scored", 0),
            spearman_rank_mean=st.get("spearman_rank_mean", np.nan),
            kendall_tau_mean=st.get("kendall_tau_mean", np.nan),
            topk_overlap_mean=st.get("topk_overlap_mean", np.nan),
            pairprob_absdiff_mean=st.get("pairprob_absdiff_mean", np.nan),
            ci_overlap_frac_mean=st.get("ci_overlap_frac_mean", np.nan),
        ))
    df = pd.DataFrame(rows)

    def rank01(s, higher_better=True):
        s = s.copy()
        nan_mask = ~np.isfinite(s.values)
        if nan_mask.any():
            fill = np.nanmin(s.values[np.isfinite(s.values)]) if higher_better else np.nanmax(s.values[np.isfinite(s.values)])
            s.values[nan_mask] = fill - 1e9 if higher_better else fill + 1e9
        r = s.rank(ascending=not higher_better, method="average")
        return 1.0 - (r - 1.0) / max(1.0, (len(r) - 1.0))

    df["rank_spearman"] = rank01(df["spearman_rank_mean"], higher_better=True)
    df["rank_rmse"] = rank01(df["rmse_logn_mean"], higher_better=False)
    df["rank_pairprob"] = rank01(df["pairprob_absdiff_mean"], higher_better=False)
    df["rank_ci"] = rank01(df["ci_overlap_frac_mean"], higher_better=True)

    df["composite_score"] = (
        0.40 * df["rank_spearman"]
        + 0.30 * df["rank_rmse"]
        + 0.20 * df["rank_pairprob"]
        + 0.10 * df["rank_ci"]
    )
    
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
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

  run_suite(sys.modules[__name__], 'count')


if __name__ == '__main__':
  main()
