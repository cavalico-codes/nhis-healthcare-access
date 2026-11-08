"""
nhis_survey.py -- helpers to replicate / correct the NHIS 2024 healthcare-access analysis in Python.

* load_adult(): reads Data/adult24csv.zip (and optionally merges nothing else)
* recode(): exact port of thesis Appendix A (SAS data step)
* svy_logit(): survey-weighted logistic regression with Taylor-linearized (design-based)
  variance for a stratified, clustered (PSTRAT / PPSU) design, as in SAS PROC SURVEYLOGISTIC.
  Supports true DOMAIN (subpopulation) estimation as well as SAS-WHERE-style subsetting.
"""
from __future__ import annotations
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

def load_adult(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "adult24csv.zip"
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        return pd.read_csv(z.open(name))

# --------------------------------------------------------------------------- Appendix A port
_MISSING_CODES = {
    "HINOTYR_A": (7, 9), "USUALPL_A": (7, 8, 9), "MEDDL12M_A": (7, 8, 9), "MEDNG12M_A": (7, 8, 9),
    "TRANSPOR_A": (7, 8, 9), "RXDG12M_A": (7, 8, 9), "COVER_A": (7,), "COVER65_A": (7,),
    "FDSCAT4_A": (8,), "HOUTENURE_A": (7, 8, 9), "SEX_A": (7, 9), "PHSTAT_A": (7, 9),
}

def recode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for v, codes in _MISSING_CODES.items():
        d.loc[d[v].isin(codes), v] = np.nan
    r = d["RATCAT_A"]
    d["POV_CAT"] = np.select([r.between(1, 7), r.between(8, 13), r == 14], [1, 2, 3], default=np.nan)
    a = d["AGEP_A"]
    d["AGE_CAT"] = np.select([a.between(18, 34), a.between(35, 49), a.between(50, 64), a >= 65], [1, 2, 3, 4], default=np.nan)
    e = d["EDUCP_A"]
    d["EDU_CAT"] = np.select([e.between(0, 2), e.between(3, 4), e.between(5, 7), e.between(8, 10)], [1, 2, 3, 4], default=np.nan)
    ra = d["RACEALLP_A"]
    d["RACE_CAT"] = np.select(
        [d["HISP_A"] == 1, ra == 1, ra == 2, ra == 3, ra.isin([4, 5]), ra.isin([6, 8])],
        [1, 2, 3, 4, 5, 6], default=np.nan)
    d["insured"] = d["COVER_A"].isin([1, 2, 3]).astype(int)      # as in thesis Appendix A
    d["age65plus"] = (d["AGEP_A"] >= 65).astype(int)
    return d

# labels / reference categories used in the thesis (SAS param=ref)
PREDICTORS = ["POV_CAT", "FDSCAT4_A", "HOUTENURE_A", "REGION", "URBRRL23", "SEX_A", "EDU_CAT", "RACE_CAT", "AGE_CAT", "PHSTAT_A"]
REFS = {"POV_CAT": 1, "FDSCAT4_A": 1, "HOUTENURE_A": 1, "REGION": 3, "URBRRL23": 1, "SEX_A": 1,
        "EDU_CAT": 1, "RACE_CAT": 2, "AGE_CAT": 3, "PHSTAT_A": 2}
LEVEL_LABELS = {
    "POV_CAT": {1: "<2 FPL", 2: "2-5 FPL", 3: ">5 FPL"},
    "FDSCAT4_A": {1: "High", 2: "Marginal", 3: "Low", 4: "Very low"},
    "HOUTENURE_A": {1: "Owned", 2: "Rented", 3: "Other"},
    "REGION": {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    "URBRRL23": {1: "Large central", 2: "Large fringe", 3: "Medium/small metro", 4: "Nonmetropolitan"},
    "SEX_A": {1: "Male", 2: "Female"},
    "EDU_CAT": {1: "Less than HS", 2: "High School", 3: "Some College", 4: "Bachelor+"},
    "RACE_CAT": {1: "Hispanic", 2: "White", 3: "Black", 4: "Asian", 5: "AIAN", 6: "Other"},
    "AGE_CAT": {1: "18-34", 2: "35-49", 3: "50-64", 4: "65+"},
    "PHSTAT_A": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
}

def design_matrix(d: pd.DataFrame, predictors=PREDICTORS, refs=REFS):
    """Dummy-code (param=ref). Returns X (with const) and a list of (var, level, ref) per column."""
    cols, meta = [pd.Series(1.0, index=d.index, name="const")], [("const", None, None)]
    for v in predictors:
        for lev in sorted(d[v].dropna().unique()):
            if lev == refs[v]:
                continue
            cols.append((d[v] == lev).astype(float).rename(f"{v}={int(lev)}"))
            meta.append((v, int(lev), refs[v]))
    return pd.concat(cols, axis=1), meta

# --------------------------------------------------------------------------- survey logistic
def _taylor_cov(X, y, p, w, strata, psu, A_inv):
    """Binder (1983) linearization: A^-1 G A^-1, stratified with-replacement PSU totals."""
    u = X * (w * (y - p))[:, None]
    strata = np.asarray(strata); psu = np.asarray(psu)
    pair = pd.MultiIndex.from_arrays([strata, psu])
    cid, uniq = pd.factorize(pair)                       # one id per (stratum, PSU)
    tot = np.zeros((len(uniq), X.shape[1]))
    np.add.at(tot, cid, u)                               # PSU-level score totals
    cl_stratum = np.array([t[0] for t in uniq])
    G = np.zeros((X.shape[1], X.shape[1]))
    for s_ in np.unique(cl_stratum):
        grp = tot[cl_stratum == s_]
        n_h = grp.shape[0]
        if n_h < 2:
            continue                                     # single-PSU strata contribute nothing
        z = grp - grp.mean(axis=0)
        G += n_h / (n_h - 1) * z.T @ z
    return A_inv @ G @ A_inv

def svy_logit(d: pd.DataFrame, y: str, event, predictors=PREDICTORS, refs=REFS,
              weight="WTFA_A", strata="PSTRAT", psu="PPSU", domain: pd.Series | None = None,
              dist="t", full_design: pd.DataFrame | None = None):
    """
    Survey-weighted logistic regression of 1{y==event} on dummy-coded predictors.

    domain=None  -> analyse all rows with complete data (== SAS without WHERE/DOMAIN).
    domain=mask  -> TRUE subpopulation analysis: rows outside the mask are retained in the design
                    (their scores are zero) so PSU/strata structure and df come from the full sample.
                    (Use this instead of subsetting; subsetting understates/misstates variance.)
    dist: "t" uses df = #PSU - #strata (SAS default for CIs); "z" uses the normal.
    Returns (results DataFrame, info dict).
    """
    need = [y, weight, strata, psu] + list(predictors)
    dd = d.dropna(subset=need).copy()
    if domain is not None:
        dmask = domain.reindex(dd.index).fillna(False).astype(bool).values
    else:
        dmask = np.ones(len(dd), dtype=bool)
    X, meta = design_matrix(dd.loc[dmask], predictors, refs)
    Xfull = X.reindex(columns=X.columns).reindex(dd.index)       # NaN outside domain
    Xfull = Xfull.fillna(0.0)
    Xv = Xfull.values.astype(float)
    yv = (dd[y].values == event).astype(float)
    wv = dd[weight].values.astype(float) * dmask               # zero weight outside domain
    # point estimates (weights outside domain are zero -> identical to domain-only fit)
    mod = sm.GLM(yv[dmask], Xv[dmask], family=sm.families.Binomial(), freq_weights=wv[dmask])
    res = mod.fit(maxiter=100, tol=1e-10)
    beta = res.params.values if hasattr(res.params, "values") else res.params
    eta = Xv @ beta
    p = 1 / (1 + np.exp(-eta))
    A = (Xv * (wv * p * (1 - p))[:, None]).T @ Xv
    A_inv = np.linalg.inv(A)
    cov = _taylor_cov(Xv, yv, p, wv, dd[strata].values, dd[psu].values, A_inv)
    se = np.sqrt(np.diag(cov))
    n_psu = dd.groupby(strata)[psu].nunique().sum() if domain is not None else dd.loc[dmask].groupby(strata)[psu].nunique().sum()
    n_str = dd[strata].nunique() if domain is not None else dd.loc[dmask, strata].nunique()
    dof = n_psu - n_str
    crit = stats.t.ppf(0.975, dof) if dist == "t" else stats.norm.ppf(0.975)
    tstat = beta / se
    pval = 2 * (stats.t.sf(np.abs(tstat), dof) if dist == "t" else stats.norm.sf(np.abs(tstat)))
    out = pd.DataFrame({
        "var": [m[0] for m in meta], "level": [m[1] for m in meta], "ref": [m[2] for m in meta],
        "beta": beta, "se": se, "OR": np.exp(beta),
        "lo": np.exp(beta - crit * se), "hi": np.exp(beta + crit * se), "p": pval})
    info = {"n": int(dmask.sum()), "n_design": len(dd), "n_events": int(yv[dmask].sum()), "df": int(dof),
            "n_psu": int(n_psu), "n_strata": int(n_str), "beta": beta, "cov": cov, "cols": list(X.columns)}
    return out, info


# --------------------------------------------------------------------------- ordinal (cumulative logit)
def svy_ordinal(d: pd.DataFrame, y: str, predictors=PREDICTORS, refs=REFS, weight="WTFA_A",
                strata="PSTRAT", psu="PPSU", dist="t"):
    """
    Survey-weighted proportional-odds (cumulative logit) model, SAS PROC SURVEYLOGISTIC default for a
    multi-level response: logit Pr(Y <= k) = alpha_k + x'beta, response levels ordered by internal value.
    A positive beta therefore means HIGHER odds of the LOWER-numbered response categories.
    Included only to reproduce what the thesis' SAS code did for USUALPL_A, COVER_A and COVER65_A.
    """
    from scipy.optimize import minimize
    need = [y, weight, strata, psu] + list(predictors)
    dd = d.dropna(subset=need).copy()
    X, meta = design_matrix(dd, predictors, refs)
    X = X.drop(columns="const"); meta = meta[1:]
    Xv = X.values.astype(float); wv = dd[weight].values.astype(float)
    levels = np.sort(dd[y].unique()); K = len(levels)
    yi = np.searchsorted(levels, dd[y].values)
    n, p = Xv.shape

    def probs(theta):
        a, b = theta[:K - 1], theta[K - 1:]
        eta = Xv @ b
        F = np.column_stack([np.zeros(n)] + [1 / (1 + np.exp(-(a[k] + eta))) for k in range(K - 1)] + [np.ones(n)])
        return F
    def unit_scores(theta):
        a, b = theta[:K - 1], theta[K - 1:]
        F = probs(theta); f = F * (1 - F)
        P = F[np.arange(n), yi + 1] - F[np.arange(n), yi]
        P = np.clip(P, 1e-300, None)
        S = np.zeros((n, K - 1 + p))
        for j in range(K - 1):            # dP/dalpha_j
            S[:, j] = (f[np.arange(n), yi + 1] * (yi + 1 == j + 1) - f[np.arange(n), yi] * (yi == j + 1)) / P
        dPdeta = f[np.arange(n), yi + 1] - f[np.arange(n), yi]
        S[:, K - 1:] = (dPdeta / P)[:, None] * Xv
        return S, np.log(P)
    def nll(theta):
        S, lp = unit_scores(theta); return -(wv * lp).sum()
    def grad(theta):
        S, lp = unit_scores(theta); return -(wv[:, None] * S).sum(axis=0)
    # starting values
    cum = np.cumsum(np.bincount(yi, weights=wv, minlength=K))[:-1] / wv.sum()
    theta0 = np.concatenate([np.log(cum / (1 - cum)), np.zeros(p)])
    opt = minimize(nll, theta0, jac=grad, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
    theta = opt.x
    # Newton polishing with a numerical Hessian (BFGS alone stops short on sparse categories)
    def _hess(th, eps=1e-5):
        mm = len(th); Hh = np.zeros((mm, mm))
        for j in range(mm):
            e = np.zeros(mm); e[j] = eps
            Hh[:, j] = (grad(th + e) - grad(th - e)) / (2 * eps)
        return (Hh + Hh.T) / 2
    for _ in range(50):
        g = grad(theta)
        if np.max(np.abs(g)) < 1e-7:
            break
        step = np.linalg.solve(_hess(theta), g)
        t_ = 1.0
        while nll(theta - t_ * step) > nll(theta) + 1e-12 and t_ > 1e-6:
            t_ /= 2
        theta = theta - t_ * step
    converged = bool(np.max(np.abs(grad(theta))) < 1e-5)
    # observed information by central differences of the gradient
    m = len(theta); H = np.zeros((m, m)); eps = 1e-5
    for j in range(m):
        e = np.zeros(m); e[j] = eps
        H[:, j] = (grad(theta + e) - grad(theta - e)) / (2 * eps)
    H = (H + H.T) / 2; A_inv = np.linalg.inv(H)
    S, _ = unit_scores(theta); U = wv[:, None] * S
    pair = pd.MultiIndex.from_arrays([dd[strata].values, dd[psu].values]); cid, uniq = pd.factorize(pair)
    tot = np.zeros((len(uniq), m)); np.add.at(tot, cid, U)
    cl_s = np.array([t[0] for t in uniq]); G = np.zeros((m, m))
    for s_ in np.unique(cl_s):
        g = tot[cl_s == s_]; nh = len(g)
        if nh > 1: z = g - g.mean(axis=0); G += nh / (nh - 1) * z.T @ z
    cov = A_inv @ G @ A_inv; se = np.sqrt(np.diag(cov))
    dof = len(uniq) - dd[strata].nunique()
    crit = stats.t.ppf(0.975, dof) if dist == "t" else 1.959964
    beta, sb = theta[K - 1:], se[K - 1:]
    pval = 2 * (stats.t.sf(np.abs(beta / sb), dof) if dist == "t" else stats.norm.sf(np.abs(beta / sb)))
    out = pd.DataFrame({"var": [m_[0] for m_ in meta], "level": [m_[1] for m_ in meta], "ref": [m_[2] for m_ in meta],
                        "beta": beta, "se": sb, "OR": np.exp(beta), "lo": np.exp(beta - crit * sb),
                        "hi": np.exp(beta + crit * sb), "p": pval})
    return out, {"n": n, "levels": [int(l) for l in levels], "df": int(dof), "converged": converged}


# =========================================================================== corrected-analysis helpers (notebook 02)
REFS["INS_CAT"] = 1
LEVEL_LABELS["INS_CAT"] = {1: "Insured", 2: "Uninsured"}

OUTCOMES = {   # column -> (short label, long label); every outcome is coded 1 = access problem, 0 = none
    "UNINS":         ("Uninsured",            "Uninsured at time of interview"),
    "ANYGAP":        ("Any coverage gap",     "Uninsured at interview or any gap in coverage in the past 12 months"),
    "NOUSUAL":       ("No usual place",       "No usual place of care"),
    "BAR_DELAY":     ("Delayed care (cost)",  "Delayed medical care because of cost, past 12 months"),
    "BAR_UNMET":     ("Unmet care (cost)",    "Needed medical care but did not get it because of cost, past 12 months"),
    "BAR_TRANSPORT": ("Transportation",       "Lack of transportation prevented medical care, past 12 months"),
    "BAR_RX":        ("Unmet Rx (cost)",      "Needed prescription medication but did not get it because of cost, past 12 months"),
}

def define_outcomes(df):
    """Corrected outcome definitions, all on the full adult sample, coded 1 = problem (event is always 1)."""
    d = df.copy()
    nc, h = d["NOTCOV_A"], d["HINOTYR_A"]
    d["UNINS"] = np.select([nc == 1, nc == 2], [1.0, 0.0], default=np.nan)
    # HINOTYR_A is asked only of adults with coverage indicated at interview (and CDC warns against using it for the
    # uninsured), so "any gap" = uninsured at interview OR (covered at interview AND a lapse in the past 12 months)
    d["ANYGAP"] = np.select([nc == 1, (nc == 2) & (h == 1), (nc == 2) & (h == 2)], [1.0, 1.0, 0.0], default=np.nan)
    d["NOUSUAL"] = np.select([d["USUALPL_A"] == 2, d["USUALPL_A"].isin([1, 3])], [1.0, 0.0], default=np.nan)
    for new, old in {"BAR_DELAY": "MEDDL12M_A", "BAR_UNMET": "MEDNG12M_A", "BAR_TRANSPORT": "TRANSPOR_A", "BAR_RX": "RXDG12M_A"}.items():
        d[new] = np.select([d[old] == 1, d[old] == 2], [1.0, 0.0], default=np.nan)
    d["INS_CAT"] = np.select([nc == 2, nc == 1], [1.0, 2.0], default=np.nan)      # covariate for sensitivity models
    return d

def svy_prop(d, flag, domain=None, weight="WTFA_A", strata="PSTRAT", psu="PPSU"):
    """Design-based proportion of `flag` (0/1, NaN excluded) within `domain`; Taylor SE, t CI (df = #PSU - #strata, full design)."""
    dd = d[[weight, strata, psu]].copy(); dd["y"] = pd.Series(flag, index=d.index)
    dd = dd[dd["y"].notna()]
    dm = np.ones(len(dd), bool) if domain is None else domain.reindex(dd.index).fillna(False).astype(bool).values
    w = dd[weight].values.astype(float) * dm; y = dd["y"].values.astype(float)
    W = w.sum(); p = (w * y).sum() / W
    cov = _taylor_cov(np.ones((len(dd), 1)), y, p, w, dd[strata].values, dd[psu].values, np.array([[1.0 / W]]))
    se = float(np.sqrt(cov[0, 0]))
    dof = dd.groupby(strata)[psu].nunique().sum() - dd[strata].nunique()
    crit = stats.t.ppf(0.975, dof)
    return {"p": p, "se": se, "lo": max(0.0, p - crit * se), "hi": min(1.0, p + crit * se), "n": int((dm).sum())}

def svy_marginal(d, y, event, var, predictors=PREDICTORS, refs=REFS, weight="WTFA_A", strata="PSTRAT", psu="PPSU", domain=None, fit=None):
    """Adjusted prevalence by level of `var` (marginal standardization over the weighted sample's covariates) with delta-method
    CIs from the survey-design covariance of the logistic coefficients, plus the adjusted risk difference vs the reference level."""
    res, info = fit if fit is not None else svy_logit(d, y, event, predictors, refs, weight, strata, psu, domain)
    need = [y, weight, strata, psu] + list(predictors)
    dd = d.dropna(subset=need)
    dm = np.ones(len(dd), bool) if domain is None else domain.reindex(dd.index).fillna(False).astype(bool).values
    X, _ = design_matrix(dd.loc[dm], predictors, refs)
    w = dd.loc[dm, weight].values.astype(float); W = w.sum()
    beta, V = info["beta"], info["cov"]
    dummies = [c for c in X.columns if c.startswith(var + "=")]
    levels = sorted(dd.loc[dm, var].unique())
    means, grads = {}, {}
    for lev in levels:
        X2 = X.copy()
        for c in dummies:
            X2[c] = 1.0 if c == f"{var}={int(lev)}" else 0.0
        Xv = X2.values
        pr = 1 / (1 + np.exp(-(Xv @ beta)))
        means[lev] = (w * pr).sum() / W
        grads[lev] = (Xv * (w * pr * (1 - pr))[:, None]).sum(0) / W
    ref = refs[var]; dof = info["df"]; crit = stats.t.ppf(0.975, dof); rows = []
    for lev in levels:
        se = float(np.sqrt(grads[lev] @ V @ grads[lev]))
        r = {"var": var, "level": int(lev), "adj_prev": means[lev], "se": se,
             "lo": max(0.0, means[lev] - crit * se), "hi": min(1.0, means[lev] + crit * se)}
        if lev != ref:
            g = grads[lev] - grads[ref]; sd = float(np.sqrt(g @ V @ g)); rd = means[lev] - means[ref]
            r.update(rd=rd, rd_lo=rd - crit * sd, rd_hi=rd + crit * sd, rd_p=2 * stats.t.sf(abs(rd / sd), dof))
        rows.append(r)
    return pd.DataFrame(rows)

def pool_mi(fits, df_com):
    """Rubin's rules with the Barnard-Rubin small-sample df (as SAS PROC MIANALYZE with EDF=). fits: list of svy_logit result frames."""
    m = len(fits)
    B_ = np.column_stack([f["beta"].values for f in fits]); U_ = np.column_stack([f["se"].values ** 2 for f in fits])
    q, ubar, b = B_.mean(1), U_.mean(1), B_.var(1, ddof=1)
    T = ubar + (1 + 1 / m) * b
    gam = (1 + 1 / m) * b / T
    with np.errstate(divide="ignore"):
        inv_nu_m = np.where(gam > 0, gam ** 2 / (m - 1), 0.0)
    nu_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - gam)
    nu = 1 / (inv_nu_m + 1 / nu_obs)
    se = np.sqrt(T); crit = stats.t.ppf(0.975, nu)
    out = fits[0][["var", "level", "ref"]].copy()
    out["beta"], out["se"], out["OR"] = q, se, np.exp(q)
    out["lo"], out["hi"] = np.exp(q - crit * se), np.exp(q + crit * se)
    out["p"] = 2 * stats.t.sf(np.abs(q / se), nu); out["fmi"] = gam
    return out

def bh_qvalues(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q
