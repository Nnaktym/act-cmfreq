"""
=============================================================================
Descriptive characterization of the VehGroup x State collision-cost matrix
=============================================================================

Produces the exposure-weighted summaries interpreted in the paper's Section 4.2
("Actual claim costs by vehicle group and state"): which states and which
vehicle groups / manufacturers carry the highest and lowest collision pure
premium, the relative size of the vehicle-group vs state main effects, and an
honest diagnostic of how much of the variation the two main effects already
explain (i.e. how much room is left for interaction + noise).

All aggregates are EXPOSURE-WEIGHTED pure premium = sum(claim) / sum(exposure)
over observed cells -- the actuarially correct way to summarise a rate, and the
same weighting the models use. Run from any directory:

    python src/data_characterization.py
"""

import os

import numpy as np
import pandas as pd

from helper import load_cell_matrix

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(_ROOT, "data", "brvehins1_full.csv")
DOCS_DIR = os.path.join(_ROOT, "docs")

MIN_CELLS = 3          # a vehicle group needs >=3 observed states to be ranked
MANU_MIN_EXPOSURE = 2e5  # a manufacturer needs this much exposure to be ranked


def _weighted_pp(claim, exp, obs, axis, labels, name):
    """Exposure-weighted pure premium collapsed along one axis of the matrix."""
    c = np.nansum(np.where(obs, claim, 0.0), axis=axis)
    e = np.nansum(np.where(obs, exp, 0.0), axis=axis)
    n = obs.sum(axis=axis)
    pp = np.where(e > 0, c / e, np.nan)
    df = pd.DataFrame({name: labels, "pp_weighted": pp, "exposure": e, "n_cells": n})
    return df[df["exposure"] > 0].sort_values("pp_weighted", ascending=False)


def _main_effect_r2(claim, exp, obs, models, areas):
    """Share of exposure-weighted pure-premium variance explained by the two
    MAIN effects alone (weighted two-way fit). 1 - R2 is the headroom left for
    the vehicle x state interaction plus cell noise. Fit on the rate scale
    (not log) so that legitimate zero-claim cells are handled correctly."""
    from statsmodels.formula.api import wls

    er, ec = np.where(obs)
    df = pd.DataFrame({
        "pp": claim[er, ec] / exp[er, ec], "w": exp[er, ec],
        "veh": models[er], "state": areas[ec],
    })
    fit = wls("pp ~ C(veh) + C(state)", data=df, weights=df["w"]).fit()
    return fit.rsquared


def main(peril="collision"):
    os.makedirs(DOCS_DIR, exist_ok=True)
    sfx = "" if peril == "collision" else f"_{peril}"
    pp_df, exp_df = load_cell_matrix(
        csv_path=DATA_CSV, brand=None, target="pure_premium",
        row_col="VehGroup", col_col="State", peril=peril)
    P = pp_df.to_numpy(float)
    E = exp_df.to_numpy(float)
    obs = ~np.isnan(P)
    claim = np.where(obs, P * E, np.nan)   # peril claim amount per cell
    models = pp_df.index.to_numpy()
    areas = pp_df.columns.to_numpy()
    print(f"peril: {peril}")

    overall = np.nansum(claim) / np.nansum(np.where(obs, E, np.nan))
    print(f"observed cells: {int(obs.sum())}  "
          f"overall exposure-weighted pure premium = {overall:,.1f}")
    print(f"per-cell pure premium: median={np.nanmedian(P):,.0f}  "
          f"p90={np.nanpercentile(P[obs], 90):,.0f}  max={np.nanmax(P):,.0f}")

    by_state = _weighted_pp(claim, E, obs, 0, areas, "State")
    by_veh = _weighted_pp(claim, E, obs, 1, models, "VehGroup")
    by_veh_ranked = by_veh[by_veh["n_cells"] >= MIN_CELLS]

    manu_labels = pd.Series(models).str.split().str[0].to_numpy()
    rows = []
    for m in np.unique(manu_labels):
        mask = manu_labels == m
        c = np.nansum(np.where(obs[mask], claim[mask], 0.0))
        e = np.nansum(np.where(obs[mask], E[mask], 0.0))
        if e > 0:
            rows.append({"Manufacturer": m, "pp_weighted": c / e,
                         "exposure": e, "n_groups": int(mask.sum())})
    by_manu = pd.DataFrame(rows).sort_values("pp_weighted", ascending=False)
    by_manu_ranked = by_manu[by_manu["exposure"] >= MANU_MIN_EXPOSURE]

    def _spread(lo, hi):
        return f"{hi/lo:,.0f}x" if lo > 0 else f">{hi:,.0f}x (min rate is 0)"

    veh_range = by_veh_ranked["pp_weighted"]
    st_range = by_state["pp_weighted"]
    print(f"\nvehicle-group spread: {veh_range.min():,.0f} .. {veh_range.max():,.0f} "
          f"({_spread(veh_range.min(), veh_range.max())})")
    print(f"state spread        : {st_range.min():,.0f} .. {st_range.max():,.0f} "
          f"({_spread(st_range.min(), st_range.max())})")

    r2 = _main_effect_r2(claim, E, obs, models, areas)
    print(f"main-effects (veh + state) weighted R^2 on pure premium: {r2:.3f} "
          f"-> {(1-r2)*100:.0f}% left for interaction + noise")

    print("\n=== STATE: highest ==="); print(by_state.head(8).to_string(index=False))
    print("--- lowest ---");          print(by_state.tail(5).to_string(index=False))
    print(f"\n=== VEHICLE GROUP (>= {MIN_CELLS} cells): highest ===")
    print(by_veh_ranked.head(12).to_string(index=False))
    print("--- lowest ---"); print(by_veh_ranked.tail(8).to_string(index=False))
    print(f"\n=== MANUFACTURER (exposure >= {MANU_MIN_EXPOSURE:,.0f}) ===")
    print(by_manu_ranked.to_string(index=False))

    by_state.to_csv(f"{DOCS_DIR}/data_char_by_state{sfx}.csv", index=False)
    by_veh.to_csv(f"{DOCS_DIR}/data_char_by_vehgroup{sfx}.csv", index=False)
    by_manu.to_csv(f"{DOCS_DIR}/data_char_by_manufacturer{sfx}.csv", index=False)
    print(f"\nsaved data_char_by_state/_by_vehgroup/_by_manufacturer{sfx}.csv to {DOCS_DIR}")


if __name__ == "__main__":
    import sys
    peril = sys.argv[1] if len(sys.argv) > 1 else "collision"
    main(peril=peril)
