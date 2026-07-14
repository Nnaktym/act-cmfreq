# archive/

Code that is **no longer part of the paper's analysis pipeline** — kept for
reference and reproducibility of earlier stages. Nothing here is imported or run
by the current Python pipeline in `src/`.

The live analysis lives entirely in `src/` (`ratemaking.py`,
`brazil_data_analysis_R.py`, `glmm_pymc.py`, `sensitivity_exposure.py`).

## Contents

```text
cmf.r                            root prototype: the very first R script
                                 (unweighted MF, weight=pt experiment)
brazil_data_analysis_python.py   earlier side-info CMF experiment (pre-refactor
                                 Python; the "Preliminary experiments (CMF)" the
                                 paper's conclusion cites as future work)
R/
  cmf.R                          original R helpers (data load/agg, split/CV,
                                 metrics, viz) — ported to src/ratemaking.py
  brazil_data_analysis_R.R       original R analysis (reference); unweighted MF
  brazil_data_analysis_R.ipynb   original R notebook (same analysis as .R)
  export_brvehins_full.R         one-off R export of the Honda raw data
```

## Why these are archived, not deleted

- **R scripts are historical/unweighted.** `cmf.R` / `brazil_data_analysis_R.R`
  fit the MF *without* exposure weighting. The exposure-weighted comparison the
  paper reports is the Python pipeline only. They document the earlier,
  unweighted reference implementation.
- **`brazil_data_analysis_python.py`** predates the `ratemaking.py` refactor and
  covers the side-information CMF experiment that the paper treats as future
  work, not a headline result.
