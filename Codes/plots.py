"""plots.py -- forest-plot panels and heatmap for the NHIS access analysis (matplotlib)."""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from nhis_survey import LEVEL_LABELS

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a85"
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"        # validated reference palette: series-1, red pole, neutral midpoint
GRID = "#e4e3df"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})

SHOW_TITLES = False          # journal figures carry their titles in the caption
PRETTY = True                # readable axis labels instead of NHIS variable codes
VAR_NAME = {"POV_CAT": "Income", "FDSCAT4_A": "Food security", "HOUTENURE_A": "Housing", "REGION": "Region",
            "URBRRL23": "Urbanicity", "SEX_A": "Sex", "EDU_CAT": "Education", "RACE_CAT": "Race/ethnicity",
            "AGE_CAT": "Age", "PHSTAT_A": "Self-rated health"}
PLEVELS = {"POV_CAT": {1: "<2\u00d7 FPL", 2: "2\u2013<5\u00d7 FPL", 3: "\u22655\u00d7 FPL"},
           "FDSCAT4_A": {1: "High", 2: "Marginal", 3: "Low", 4: "Very low"},
           "HOUTENURE_A": {1: "Owned", 2: "Rented", 3: "Other"},
           "REGION": {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
           "URBRRL23": {1: "Large central", 2: "Large fringe", 3: "Medium/small metro", 4: "Nonmetro"},
           "SEX_A": {1: "Male", 2: "Female"},
           "EDU_CAT": {1: "<High school", 2: "High school", 3: "Some college", 4: "Bachelor+"},
           "RACE_CAT": {1: "Hispanic", 2: "NH White", 3: "NH Black", 4: "NH Asian", 5: "NH AIAN", 6: "NH other"},
           "AGE_CAT": {1: "18\u201334", 2: "35\u201349", 3: "50\u201364", 4: "\u226565"},
           "PHSTAT_A": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}}

def row_label(var, level, ref):
    if PRETTY:
        L = PLEVELS[var]; return f"{VAR_NAME[var]}: {L[level]} vs {L[ref]}"
    L = LEVEL_LABELS[var]
    return f"{var}: {L[level]} vs {L[ref]}"

def _clean(df):
    """Drop the intercept row and make level codes integers so labels can be looked up."""
    df = df[df["var"] != "const"].reset_index(drop=True).copy()
    df["level"] = df["level"].astype(int); df["ref"] = df["ref"].astype(int)
    return df

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def forest_panels(results: dict, titles: dict | None = None, ncols=3, xlim=(0.1, 10), suptitle=None, outfile=None,
                  xlabel="Adjusted odds ratio (log scale)", figsize=(15, 11)):
    results = {k: _clean(v) for k, v in results.items()}
    names = list(results)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    first = results[names[0]]
    labels = [row_label(v, l, r) for v, l, r in zip(first["var"], first["level"], first["ref"])]
    y = np.arange(len(labels))[::-1]
    for ax, nm in zip(axes, names):
        df = results[nm]
        sig = df["p"].values < 0.05
        col = np.where(sig, BLUE, MUTED)
        for yi, o, lo, hi, c, p in zip(y, df["OR"], df["lo"], df["hi"], col, df["p"]):
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [yi, yi], color=c, lw=1.3, solid_capstyle="round", zorder=2)
            ax.scatter([o], [yi], s=26, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.text(xlim[1] * 1.04, yi, stars(p), va="center", ha="left", color=INK2, fontsize=6.5, clip_on=False)
        ax.axvline(1, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log"); ax.set_xlim(*xlim)
        tk = [t for t in (0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50) if xlim[0] <= t <= xlim[1]]
        ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk])
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title((titles or {}).get(nm, nm), loc="left", fontsize=8.5, color=INK, fontweight="bold")
    for ax in axes[len(names):]: ax.axis("off")
    for ax in axes[:len(names)]: ax.set_xlabel(xlabel, fontsize=7.5)
    handles = [Line2D([0], [0], marker="o", color=BLUE, lw=1.3, label="p < 0.05"),
               Line2D([0], [0], marker="o", color=MUTED, lw=1.3, label="p ≥ 0.05")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    if suptitle and SHOW_TITLES: fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97 if (suptitle and SHOW_TITLES) else 1))
    if outfile: fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig

def heatmap(results: dict, titles: dict | None = None, vlim=1.8, outfile=None, suptitle=None, figsize=(8.5, 9.5),
            cbar_label="log(adjusted odds ratio)"):
    results = {k: _clean(v) for k, v in results.items()}
    names = list(results)
    first = results[names[0]]
    labels = [row_label(v, l, r) for v, l, r in zip(first["var"], first["level"], first["ref"])]
    M = np.column_stack([np.log(results[n]["OR"].values) for n in names])
    P = np.column_stack([results[n]["p"].values for n in names])
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, stars(P[i, j]), ha="center", va="center", color=INK, fontsize=7)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([(titles or {}).get(n, n) for n in names], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True); ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5); ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both"); cb.set_label(cbar_label, fontsize=7.5); cb.outline.set_visible(False)
    if suptitle and SHOW_TITLES: ax.set_title(suptitle, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    if outfile: fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig


def auto_xlim(frames):
    lo = min(f["lo"].min() for f in frames); hi = max(f["hi"].max() for f in frames)
    return (0.02 if lo < 0.05 else 0.05 if lo < 0.1 else 0.1, 50 if hi > 20 else 20 if hi > 10 else 10)

def prev_panels(margins: dict, var, titles=None, ncols=4, outfile=None, suptitle=None, figsize=(10, 5.6),
                ylabel="Adjusted prevalence (%)"):
    """Small multiples of covariate-adjusted prevalence (marginal standardization) by level of one determinant."""
    names = list(margins); nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel(); L = LEVEL_LABELS[var]
    for ax, nm in zip(axes, names):
        m = margins[nm]; x = np.arange(len(m))
        ax.vlines(x, 100 * m["lo"], 100 * m["hi"], color=BLUE, lw=1.8, zorder=2)
        ax.scatter(x, 100 * m["adj_prev"], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for xi, v, hi in ((x[0], m["adj_prev"].iloc[0], m["hi"].iloc[0]), (x[-1], m["adj_prev"].iloc[-1], m["hi"].iloc[-1])):
            ax.text(xi, 100 * hi + 0.02 * 100 * m["hi"].max(), f"{100 * v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([PLEVELS[var][int(l)] if PRETTY else L[int(l)] for l in m["level"]], fontsize=7)
        ax.set_xlim(-0.5, len(m) - 0.5); ax.set_ylim(0, 100 * m["hi"].max() * 1.18)
        ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_title((titles or {}).get(nm, nm), loc="left", fontsize=8.5, fontweight="bold", color=INK)
    for ax in axes[len(names):]: ax.axis("off")
    for ax in axes[::ncols]: ax.set_ylabel(ylabel, fontsize=7.5)
    if suptitle and SHOW_TITLES: fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if suptitle else 1))
    if outfile: fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig
