"""Sensitivity analysis for the cell exposure threshold (cell_exposure_min).

The main analysis keeps cells whose total exposure >= 100. That threshold is a
credibility floor (exogenous: exposure is the denominator, independent of the
claim outcome), but the value 100 is a modelling choice. This script re-runs the
full split-before-tune, exposure-weighted comparison at cell_exposure_min in
{50, 100, 200} for both targets (collision pure premium and collision frequency)
and tabulates whether the ranking is robust to the choice.

It uses run_comparison(..., write=False) so it never overwrites the canonical
docs/figures. Output: docs/sensitivity_exposure.csv (overall + stratified rows).

Run:  python src/sensitivity_exposure.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brazil_data_analysis_R import prepare_data, run_comparison

THRESHOLDS = [50, 100, 200]
TARGETS = ["pure_premium", "frequency"]


def main():
    os.makedirs("docs", exist_ok=True)
    rows = []
    for target in TARGETS:
        for thr in THRESHOLDS:
            print(f"\n########## target={target}  cell_exposure_min={thr} ##########")
            pp, pp_mat, exp_mat, obs, W, U, I = prepare_data(
                target=target, cell_exposure_min=thr)
            best, ctx = run_comparison(pp, pp_mat, exp_mat, W, U, I,
                                       target=target, write=False)
            n_obs = int(obs.sum())
            n_eval = int(ctx["n_eval"])
            comp = ctx["comparison"]
            for model in comp.index:
                rows.append({
                    "target": target, "cell_exposure_min": thr,
                    "n_obs": n_obs, "n_eval": n_eval, "stratum": "all",
                    "model": model,
                    "RMSE": comp.loc[model, "RMSE"],
                    "wRMSE": comp.loc[model, "wRMSE(exposure)"],
                    "PoissonDev": comp.loc[model, "PoissonDeviance"],
                })
            strat = ctx["strat"]
            for _, r in strat.iterrows():
                rows.append({
                    "target": target, "cell_exposure_min": thr,
                    "n_obs": n_obs, "n_eval": n_eval,
                    "stratum": r["stratum"].strip(), "model": r["model"],
                    "RMSE": float("nan"),
                    "wRMSE": r["wRMSE"], "PoissonDev": r["PoissonDev"],
                })

    out = pd.DataFrame(rows)
    out.to_csv("docs/sensitivity_exposure.csv", index=False)
    print("\nsaved docs/sensitivity_exposure.csv")

    # compact overall-stratum view for quick reading
    print("\n================ SENSITIVITY (overall stratum) ================")
    ov = out[out["stratum"] == "all"]
    for target in TARGETS:
        print(f"\n--- {target} ---")
        for thr in THRESHOLDS:
            sub = ov[(ov["target"] == target) & (ov["cell_exposure_min"] == thr)]
            n_obs = int(sub["n_obs"].iloc[0])
            n_eval = int(sub["n_eval"].iloc[0])
            print(f"cell_exposure_min={thr}  n_obs={n_obs}  n_eval={n_eval}")
            print(sub[["model", "RMSE", "wRMSE", "PoissonDev"]].to_string(index=False))


if __name__ == "__main__":
    main()
