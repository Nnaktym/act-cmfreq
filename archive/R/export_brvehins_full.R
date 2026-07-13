#!/usr/bin/env Rscript
# Regenerate data/brvehins1_full.csv: the FULL multi-manufacturer Brazilian
# auto-insurance dataset (CASdatasets brvehins1, all brands -- not the Honda
# subset in brvehins_org.csv). ~1.97M records, 23 columns. The CSV is git-ignored
# (~328 MB); run this once locally to (re)create it.
#
# CASdatasets is off-CRAN; install from the maintainer's repository:
if (!requireNamespace("CASdatasets", quietly = TRUE)) {
  options(timeout = 1200)
  for (p in c("xts", "zoo", "sp"))
    if (!requireNamespace(p, quietly = TRUE))
      install.packages(p, repos = "https://cloud.r-project.org")
  install.packages("CASdatasets",
    repos = "https://dutangc.perso.math.cnrs.fr/RRepository/pub/",
    type = "source")
}

suppressMessages(library(CASdatasets))
# brvehins1 is shipped row-split into five shards a..e
parts <- paste0("brvehins1", letters[1:5])
for (p in parts) data(list = p, package = "CASdatasets")
df <- do.call(rbind, lapply(parts, get))

cat("combined nrow:", nrow(df), " ncol:", ncol(df), "\n")
write.csv(df, "data/brvehins1_full.csv", row.names = FALSE, na = "NA")
cat("WROTE data/brvehins1_full.csv (",
    round(file.info("data/brvehins1_full.csv")$size / 1e6, 1), "MB )\n")
