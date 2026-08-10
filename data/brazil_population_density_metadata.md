# Metadata — `brazil_population_density.csv`

Population-density indicator for the `Area` regions in `brvehins_org.csv`, used as
an urban/rural proxy (column side information for the Collective MF model).

- **File:** `data/brazil_population_density.csv`
- **Rows:** 40 (one per distinct `Area` in `brvehins_org.csv`)
- **Created:** 2026-07-12
- **Unit:** inhabitants per km² (hab/km²)

## Source

| | |
|---|---|
| **Primary source** | IBGE — Censo Demográfico 2022 (demographic density by federative unit) |
| **Retrieved via** | Wikipedia, *List of Brazilian states by population density*, "Density 2022" column (values attributed to IBGE Censo 2022) |
| **URL** | https://en.wikipedia.org/wiki/List_of_Brazilian_states_by_population_density |
| **Access date** | 2026-07-12 |
| **Geographic level** | State (federative unit) — 27 units |

## Method

The `brvehins_org` regions (`Area`) are a mix of whole states and custom insurer
sub-state groupings. There is **no official density for the custom groupings**, so
per the chosen approach each `Area` is assigned its **parent state's** IBGE 2022
density (the "parent-state density" option).

1. Look up IBGE 2022 density for each of the 27 states.
2. Map every `Area` → its parent `State` (from `brvehins_org.csv`) → state density.
3. **Blended cell:** the label `"Demais regioes"` is shared by *Paraná* and *Rio
   Grande do Sul*. The analysis matrix already merges them into one column (pivot is
   on `Area` alone), so its density is the **exposure-weighted mean** of the two
   states' densities: (43.46·ExpoPR + 36.84·ExpoRS)/(ExpoPR+ExpoRS) = **39.96**.

## Column dictionary (`brazil_population_density.csv`)

| Column | Description |
|---|---|
| `Area` | Region label; matches the `Area` values / matrix columns in `brvehins_org.csv` |
| `density_km2` | Population density, hab/km² (parent state's IBGE 2022 value; blended for the shared label) |
| `parent_state` | State the density was taken from; `A+B` for a blended cell |
| `source` | `IBGE Censo 2022` |
| `note` | `single-state` or `blended (exposure-weighted)` |

## Coverage / missingness

| | count | exposure share |
|---|---|---|
| Clean single-state density | 39 / 40 | 96.6% |
| Blended (label spans 2 states) | 1 / 40 | 3.4% |
| **Truly missing** | **0** | **0%** |

## Known limitations

- **Within-state contrast is collapsed.** Sub-state groupings inherit one state-wide
  value, so metro vs. interior distinctions are lost. This yields some counterintuitive
  urban/rural labels at the pipeline's 0.7-quantile threshold (79.4 hab/km²), e.g.
  `Met. Curitiba` → *rural* (Paraná state density 43.46) and `Interior` (Rio de Janeiro)
  → *urban* (RJ state density 387.46). To recover the metro/interior signal, densities
  would need to be built by municipal aggregation instead.
- **Blended cell** (`Demais regioes`) is a mixture, not a single official figure.
- State densities are 2022-Census; the insurance portfolio predates that, so this is a
  contemporary proxy for relative urbanicity, not a period-matched measurement.

## Consumed by

`build_side_info()` in `src/ratemaking.py` — thresholds `density_km2` at the 0.7
quantile into `urban`/`rural` one-hot column side information for the CMF model.

## State-level source values (IBGE Censo 2022, hab/km²)

Acre 6.34 · Alagoas 125.52 · Amapá 2.63 · Amazonas 2.58 · Bahia 30.52 · Ceará 60.33 ·
Distrito Federal 493.00 · Espírito Santo 80.63 · Goiás 18.46 · Maranhão 19.03 ·
Mato Grosso 4.01 · Mato Grosso do Sul 7.83 · Minas Gerais 31.72 · Pará 7.02 ·
Paraíba 78.93 · Paraná 43.46 · Pernambuco 103.83 · Piauí 9.73 · Rio de Janeiro 387.46 ·
Rio Grande do Norte 62.74 · Rio Grande do Sul 36.84 · Rondônia 7.34 · Roraima 2.54 ·
Santa Catarina 69.74 · São Paulo 175.73 · Sergipe 97.64 · Tocantins 5.74

## Provenance note

This file replaced a prior, undocumented `brazil_population_density.csv` (Area-level
metro densities, no cited source). The previous version remains recoverable via git:
`git show HEAD:data/brazil_population_density.csv`.
