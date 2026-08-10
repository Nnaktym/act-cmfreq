# Theft (robbery) exposure in brvehins1 — data note

**Finding.** The CASdatasets `brvehins1` dataset defines a dedicated fire+theft
exposure column, `ExposFireRob` ("Exposure for fire and robbery guarantees"),
but ships it **all-zero in every shard**. The paired premium column
`PremFireRob` is likewise all-zero. The robbery *claims* (`ClaimAmountRob`,
`ClaimNbRob`) are populated (≈1.47 bn total claim amount, ~2.2% of records).

**Verification.**
- Read the raw shard `brvehins1a.rda` directly (via the pure-Python `rdata`
  reader, no R): all 393,071 rows have `ExposFireRob == 0` and `PremFireRob == 0`.
- The combined CSV `data/brvehins1_full.csv` (1,965,355 rows = shards a–e) has
  `ExposFireRob.sum() == 0` and `PremFireRob.sum() == 0`.
- So the empty column is a property of the **source data**, not of our CSV
  export; there is no fire/theft exposure to recover from the shards.

Docs: <https://dutangc.github.io/CASdatasets/reference/brvehins.html>.

**Decision.** Theft is modelled against `ExposTotal` — the comprehensive-cover
exposure that every peril's claims (collision, theft, fire, other) arise from in
this AUTOSEG extract. So the theft target is

    theft pure premium = ClaimAmountRob / ExposTotal
    theft frequency    = ClaimNbRob     / ExposTotal

**Caveat.** `ExposTotal` is not a theft-*specific* exposure. If some policies
carried fire/theft cover without collision (or vice versa), `ExposTotal` would
misstate the truly theft-exposed base — but the data records no such split
(`ExposFireRob` being globally zero), so `ExposTotal` is the only defensible
common denominator and is comparable across all cells.

Implemented as `peril="theft"` in `helper.load_cell_matrix`
(`src/helper.py`), reproducible via `src/data_characterization.py --peril theft`
and `src/brazil_data_analysis_R.py theft`.
