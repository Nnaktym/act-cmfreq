# archive/

Code that is **no longer part of the paper's analysis pipeline** — kept for
reference and reproducibility of earlier stages. Nothing here is imported or run
by the current Python pipeline in `src/`.

The live analysis lives entirely in `src/` (`helper.py`,
`brazil_data_analysis_R.py`, `glmm_pymc.py`) — the VehGroup × State (pure
premium) mainline.

## Contents

```text
cmf.r                            root prototype: the very first R script
                                 (unweighted MF, weight=pt experiment)
brazil_data_analysis_python.py   earlier side-info CMF experiment (pre-refactor
                                 Python; the "Preliminary experiments (CMF)" the
                                 paper's conclusion cites as future work)
vehgroup_variant.py              the VehGroup-row exploration that became the
                                 mainline; superseded once VehGroup × State was
                                 adopted as the headline analysis
vehgroup_state_validation.py     robustness checks for the VehGroup × State
                                 variant (k-vs-error curve, masked-cell recovery,
                                 multi-seed stability)
stdrel_variant.py                demographic-standardized-relativity robustness
                                 variant; its finding is documented in the paper
                                 as a limitation (demographic confounding)
sensitivity_exposure.py          exposure-threshold (cell_exposure_min 50/100/200)
                                 sensitivity / parameter study
R/
  cmf.R                          original R helpers (data load/agg, split/CV,
                                 metrics, viz) — ported to src/helper.py
  brazil_data_analysis_R.R       original R analysis (reference); unweighted MF
  brazil_data_analysis_R.ipynb   original R notebook (same analysis as .R)
  export_brvehins_full.R         one-off R export of the Honda raw data
```

## Why these are archived, not deleted

- **R scripts are historical/unweighted.** `cmf.R` / `brazil_data_analysis_R.R`
  fit the MF *without* exposure weighting. The exposure-weighted comparison the
  paper reports is the Python pipeline only. They document the earlier,
  unweighted reference implementation.
- **`brazil_data_analysis_python.py`** predates the `helper.py` (formerly
  `ratemaking.py`) refactor and
  covers the side-information CMF experiment that the paper treats as future
  work, not a headline result.
- **Variant / parameter-study scripts** (`vehgroup_variant.py`,
  `vehgroup_state_validation.py`, `stdrel_variant.py`, `sensitivity_exposure.py`)
  were the exploration and robustness work behind adopting VehGroup × State (pure
  premium) as the mainline. Their conclusions are baked into the paper; the
  final-form pipeline in `src/` no longer needs them. They import from `src/`
  (e.g. `from helper import ...`), so to re-run one, add `src/` to the path.
