"""Improved paper figures for the VehGroup x State pure-premium surface.

Rationale (see docs/heatmap_alternatives.md for the full comparison): the original
`visualize_heatmap` renders 231 vehicle-group rows in arbitrary order on a linear
colour scale with white == missing, which hides the paper's central finding (the
dominant axis is the vehicle group, with a ~4000x spread, versus ~3x across states)
and makes the actual-vs-model comparison impossible to do by eye.

This module regenerates the surface figures with four fixes, all reusing the
mainline data pipeline (`prepare_data`):

    B  seriated_actual      -- both axes ordered by marginal cost, log colour,
                               missing cells shown as an explicit off-ramp grey
    C  surface_panels       -- Actual | GLM | MF on ONE shared colour scale and the
                               SAME row/col order, so cells compare directly
    D  interaction_panels   -- each surface minus its OWN additive (row+col) fit, on
                               a diverging scale: the GLM is flat white by
                               construction, the MF carries structure
    E  marginals            -- sorted vehicle-group and state marginals (the 4000x
                               vs ~3x story, told without a heatmap)
    F  labeled_subset       -- the 12 highest + 12 lowest vehicle groups, rows
                               labelled so they are actually readable

Run:  python src/paper_heatmaps_v2.py
Out:  figs/paper_v2/
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from brazil_data_analysis_R import (  # noqa: E402
    prepare_data, _fit_weighted_mf, _fit_glm, _predict_glm, _long_frame,
)

OUT_DIR = os.path.join(_ROOT, "figs", "paper_v2")
MISSING_COLOR = "#d9d9d9"
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"
BEST_MF = {"k": 2, "lambda": 0.1}


# --------------------------------------------------------------------------- #
# Data / models
# --------------------------------------------------------------------------- #
def build_surfaces():
    """Return everything the figures need: matrices, labels, seriation order."""
    pp, pp_mat, exp_mat, obs, W_full, _U, _I = prepare_data()
    models = np.asarray(pp.index)
    areas = np.asarray(pp.columns)
    n_row, n_col = pp_mat.shape

    # MF reconstruction at the tuned rank.
    mf = _fit_weighted_mf(np.where(obs, pp_mat, 0.0), W_full, BEST_MF)
    mf_hat = np.clip(mf.A_ @ mf.B_.T, 0.0, None)

    # Main-effects GLM predicted over the full grid.
    ro, co = np.where(obs)
    glm = _fit_glm(_long_frame(models, areas, ro, co, pp_mat, exp_mat), "pure_premium")
    gr, gc = np.meshgrid(np.arange(n_row), np.arange(n_col), indexing="ij")
    grid = _long_frame(models, areas, gr.ravel(), gc.ravel(),
                       np.nan_to_num(pp_mat), np.where(exp_mat > 0, exp_mat, 1.0))
    glm_hat = _predict_glm(glm, grid, "pure_premium").reshape(n_row, n_col)

    # Seriation: order by exposure-weighted marginal cost (low -> high).
    row_eff = _wmean(pp_mat, exp_mat, obs, axis=1)
    col_eff = _wmean(pp_mat, exp_mat, obs, axis=0)
    row_order = np.argsort(np.where(np.isnan(row_eff), -1.0, row_eff))
    col_order = np.argsort(np.where(np.isnan(col_eff), -1.0, col_eff))

    return dict(
        models=models, areas=areas, n_row=n_row, n_col=n_col,
        pp_mat=pp_mat, exp_mat=exp_mat, obs=obs, mf_hat=mf_hat, glm_hat=glm_hat,
        row_eff=row_eff, col_eff=col_eff, row_order=row_order, col_order=col_order,
        vmin=max(30.0, float(np.nanpercentile(pp_mat[obs], 2))),
        vmax=float(np.nanpercentile(pp_mat[obs], 99)),
    )


def _wmean(mat, w, obs, axis):
    """Exposure-weighted mean over observed cells along `axis`."""
    m = np.where(obs, mat, 0.0)
    ww = np.where(obs, w, 0.0)
    num = np.sum(m * ww, axis=axis)
    den = np.sum(ww, axis=axis)
    return np.where(den > 0, num / den, np.nan)


def _two_way_residual(mat, obs, w, n_iter=25):
    """Ratio of a surface to its own weighted log-additive (row+col) fit.

    Returns exp(log v - a_i - b_j - c): 1.0 == exactly explained by main effects,
    so the residual isolates the vehicle-group x state interaction. A log-link
    main-effects GLM surface is additive in log space by construction and so maps
    to ~1 everywhere; only genuine interaction departs from white.
    """
    with np.errstate(divide="ignore"):
        logv = np.where(obs & (mat > 0), np.log(mat), np.nan)
    ww = np.where(obs & (mat > 0), w, 0.0)
    a = np.zeros(mat.shape[0])
    b = np.zeros(mat.shape[1])
    for _ in range(n_iter):
        ra = logv - b[None, :]
        a = np.nansum(np.where(np.isnan(ra), 0.0, ra * ww), axis=1) / np.where(
            np.nansum(np.where(np.isnan(ra), 0.0, ww), axis=1) > 0,
            np.nansum(np.where(np.isnan(ra), 0.0, ww), axis=1), 1.0)
        rb = logv - a[:, None]
        b = np.nansum(np.where(np.isnan(rb), 0.0, rb * ww), axis=0) / np.where(
            np.nansum(np.where(np.isnan(rb), 0.0, ww), axis=0) > 0,
            np.nansum(np.where(np.isnan(rb), 0.0, ww), axis=0), 1.0)
    resid = logv - a[:, None] - b[None, :]
    return np.exp(resid)


# --------------------------------------------------------------------------- #
# Shared drawing helpers
# --------------------------------------------------------------------------- #
def _seq_cmap():
    c = plt.get_cmap(SEQ_CMAP).copy()
    c.set_bad(MISSING_COLOR)
    return c


def _div_cmap():
    c = plt.get_cmap(DIV_CMAP).copy()
    c.set_bad(MISSING_COLOR)
    return c


def _reorder(mat, s):
    return mat[np.ix_(s["row_order"], s["col_order"])]


def _draw_surface(ax, mat, obs, s, norm, cmap):
    """imshow a (already reordered) surface with missing cells masked to grey."""
    disp = np.clip(mat, norm.vmin, norm.vmax)
    disp = np.ma.masked_where(~obs, disp)
    im = ax.imshow(disp, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(s["n_col"]))
    ax.set_xticklabels(s["areas"][s["col_order"]], rotation=90, fontsize=5)
    ax.set_yticks([])
    return im


def _missing_legend(ax):
    ax.legend(handles=[Patch(facecolor=MISSING_COLOR, edgecolor="none",
                             label="no data (low exposure)")],
              loc="lower right", fontsize=6, framealpha=0.9)


# --------------------------------------------------------------------------- #
# B -- seriated actual
# --------------------------------------------------------------------------- #
def fig_seriated_actual(s, path):
    norm = LogNorm(vmin=s["vmin"], vmax=s["vmax"])
    fig, ax = plt.subplots(figsize=(7.5, 9))
    obs_s = _reorder(s["obs"], s)
    _draw_surface(ax, _reorder(s["pp_mat"], s), obs_s, s, norm, _seq_cmap())
    ax.set_title("Actual pure premium — vehicle groups and states ordered by cost")
    ax.set_xlabel("State  (low → high cost)")
    ax.set_ylabel(f"Vehicle Group (n={s['n_row']})  (low → high cost)")
    _missing_legend(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=_seq_cmap()),
                 ax=ax, label="Pure Premium (log scale)")
    _save(fig, path)


# --------------------------------------------------------------------------- #
# C -- Actual | GLM | MF, shared scale
# --------------------------------------------------------------------------- #
def fig_surface_panels(s, path):
    norm = LogNorm(vmin=s["vmin"], vmax=s["vmax"])
    cmap = _seq_cmap()
    obs_s = _reorder(s["obs"], s)
    full = np.ones_like(obs_s, dtype=bool)
    panels = [("Actual (observed cells)", _reorder(s["pp_mat"], s), obs_s),
              ("Main-effects GLM (all cells)", _reorder(s["glm_hat"], s), full),
              ("Matrix factorization, k=2 (all cells)", _reorder(s["mf_hat"], s), full)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 8.5), sharey=True)
    for ax, (name, mat, obs) in zip(axes, panels):
        im = _draw_surface(ax, mat, obs, s, norm, cmap)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("State")
    axes[0].set_ylabel(f"Vehicle Group (n={s['n_row']})  (low → high cost)")
    _missing_legend(axes[0])
    fig.colorbar(im, ax=axes, label="Pure Premium (log scale)", shrink=0.55,
                 pad=0.02)
    fig.suptitle("Actual vs. predicted pure-premium surface "
                 "(shared colour scale, same row/column order)", y=0.98)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# D -- interaction (surface / own additive fit)
# --------------------------------------------------------------------------- #
def fig_interaction_panels(s, path):
    norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=2.5)
    cmap = _div_cmap()
    obs_s = _reorder(s["obs"], s)
    full = np.ones_like(obs_s, dtype=bool)
    w = s["exp_mat"]
    act = _two_way_residual(s["pp_mat"], s["obs"], w)
    glm = _two_way_residual(s["glm_hat"], full, w)
    mf = _two_way_residual(s["mf_hat"], full, w)
    panels = [("Actual", _reorder(act, s), obs_s),
              ("Main-effects GLM", _reorder(glm, s), full),
              ("Matrix factorization, k=2", _reorder(mf, s), full)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 8.5), sharey=True)
    for ax, (name, mat, obs) in zip(axes, panels):
        disp = np.ma.masked_where(~obs, mat)
        im = ax.imshow(disp, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(s["n_col"]))
        ax.set_xticklabels(s["areas"][s["col_order"]], rotation=90, fontsize=5)
        ax.set_yticks([]); ax.set_xlabel("State")
    axes[0].set_ylabel(f"Vehicle Group (n={s['n_row']})  (low → high cost)")
    fig.colorbar(im, ax=axes, label="value ÷ own additive fit", shrink=0.55, pad=0.02)
    fig.suptitle("Vehicle-group × state interaction "
                 "(red = above additive, blue = below, white = none). "
                 "The GLM is white by construction; the MF carries structure.",
                 y=0.98, fontsize=11)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# E -- marginals
# --------------------------------------------------------------------------- #
def fig_marginals(s, path):
    re = np.sort(s["row_eff"][~np.isnan(s["row_eff"])])
    ce = pd.Series(s["col_eff"], index=s["areas"]).dropna().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].plot(re, np.arange(len(re)), lw=2, color="#3b6fb0")
    axes[0].set_title(f"Vehicle-group effect  "
                      f"({re.min():.0f}–{re.max():.0f}, ~{re.max()/max(re.min(),1):.0f}× spread)")
    axes[0].set_xlabel("Exposure-weighted pure premium")
    axes[0].set_ylabel("Vehicle group (sorted, n=%d)" % len(re))
    axes[1].barh(range(len(ce)), ce.values, color="#3b6fb0")
    axes[1].set_yticks(range(len(ce))); axes[1].set_yticklabels(ce.index, fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_title(f"State effect  "
                      f"({ce.min():.0f}–{ce.max():.0f}, ~{ce.max()/ce.min():.1f}× spread)")
    axes[1].set_xlabel("Exposure-weighted pure premium")
    fig.suptitle("The dominant axis of variation is the vehicle group, not the state",
                 y=1.0)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# F -- labeled subset
# --------------------------------------------------------------------------- #
def fig_labeled_subset(s, path, n_each=12):
    valid = np.where(~np.isnan(s["row_eff"]))[0]
    order = valid[np.argsort(s["row_eff"][valid])]
    pick = np.concatenate([order[-n_each:][::-1], order[:n_each]])
    sub = s["pp_mat"][np.ix_(pick, s["col_order"])]
    sub_obs = s["obs"][np.ix_(pick, s["col_order"])]
    norm = LogNorm(vmin=s["vmin"], vmax=s["vmax"])
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = np.ma.masked_where(~sub_obs, np.clip(sub, norm.vmin, norm.vmax))
    im = ax.imshow(disp, aspect="auto", cmap=_seq_cmap(), norm=norm)
    ax.set_yticks(range(len(pick)))
    ax.set_yticklabels([str(s["models"][i])[:40] for i in pick], fontsize=6)
    ax.set_xticks(range(s["n_col"]))
    ax.set_xticklabels(s["areas"][s["col_order"]], rotation=90, fontsize=6)
    ax.axhline(n_each - 0.5, color="k", lw=1)
    ax.set_title(f"{n_each} highest + {n_each} lowest vehicle groups by cost")
    ax.set_xlabel("State  (low → high cost)")
    _missing_legend(ax)
    fig.colorbar(im, ax=ax, label="Pure Premium (log scale)")
    _save(fig, path)


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = build_surfaces()
    print(f"matrix {s['pp_mat'].shape}, observed {int(s['obs'].sum())}, "
          f"colour range [{s['vmin']:.0f}, {s['vmax']:.0f}]")
    fig_seriated_actual(s, os.path.join(OUT_DIR, "B_seriated_actual.png"))
    fig_surface_panels(s, os.path.join(OUT_DIR, "C_surface_panels.png"))
    fig_interaction_panels(s, os.path.join(OUT_DIR, "D_interaction_panels.png"))
    fig_marginals(s, os.path.join(OUT_DIR, "E_marginals.png"))
    fig_labeled_subset(s, os.path.join(OUT_DIR, "F_labeled_subset.png"))


if __name__ == "__main__":
    main()
