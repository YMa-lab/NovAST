#!/usr/bin/env Rscript
# run_singler.R
#
# Runs SingleR cell-type annotation on preprocessed benchmark data.
# Reads .mtx + .csv files prepared by singler_io.py (no Python dependency
# at runtime — avoids reticulate/basilisk compatibility issues).
#
# Usage:
#   python singler_io.py prepare --savedir <dir>    # before R
#   Rscript run_singler.R --savedir <dir>
#   python singler_io.py convert --savedir <dir>    # after R
#
# NOTE: SingleR is deterministic -- it is classified ONCE, there are no seeds.
# `--rounds` below is metadata only (recorded in SingleR_stats.json); it does not
# repeat the classification. singler_io.py takes ONLY --savedir -- passing
# --rounds to it is an error.
#
# Requirements:
#   R packages: SingleR, SingleCellExperiment, scuttle, jsonlite, optparse, Matrix

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(SingleCellExperiment)
  library(scuttle)
  library(SingleR)
  library(Matrix)
})

# ── CLI arguments ──────────────────────────────────────────────────────
option_list <- list(
  make_option("--savedir", type = "character", default = NULL,
              help = "Path to the dataset results directory (contains preprocess_args.json)"),
  make_option("--rounds", type = "integer", default = 10,
              help = "Number of seed rounds [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$savedir)) stop("--savedir is required")
savedir <- opt$savedir
rounds  <- opt$rounds

# ── Load preprocess args ───────────────────────────────────────────────
pre_args_path <- file.path(savedir, "preprocess_args.json")
if (!file.exists(pre_args_path)) stop("preprocess_args.json not found in ", savedir)
pre_args <- fromJSON(pre_args_path)

celltype_reference <- pre_args$celltype_name_reference
celltype_target  <- pre_args$celltype_name_target

cat("savedir:", savedir, "\n")
cat("celltype_reference:", celltype_reference, "\n")
cat("celltype_target:", celltype_target, "\n")

# ── Helper: read .mtx + metadata prepared by singler_io.py ────────────
read_mtx_to_sce <- function(input_dir, tag, celltype_col) {
  mat   <- readMM(file.path(input_dir, paste0(tag, ".mtx")))
  genes <- readLines(file.path(input_dir, paste0(tag, "_genes.csv")))
  cells <- readLines(file.path(input_dir, paste0(tag, "_barcodes.csv")))
  obs   <- read.csv(file.path(input_dir, paste0(tag, "_obs.csv")),
                     check.names = FALSE, colClasses = "character")
  rownames(obs) <- obs[[1]]
  obs <- obs[, -1, drop = FALSE]

  # mat is genes x cells (written transposed by singler_io.py)
  mat <- as(mat, "dgCMatrix")
  rownames(mat) <- genes
  colnames(mat) <- cells
  rownames(obs) <- cells

  SingleCellExperiment(
    assays  = list(counts = mat),
    colData = obs
  )
}

# ── Read data from .mtx + .csv ────────────────────────────────────────
input_dir <- file.path(savedir, "SingleR_input")
if (!dir.exists(input_dir)) {
  stop("SingleR_input/ not found. Run: python singler_io.py prepare --savedir ", savedir)
}

cat("Reading train data ...\n")
sce_reference <- read_mtx_to_sce(input_dir, "reference", celltype_reference)

cat("Reading test data ...\n")
sce_target <- read_mtx_to_sce(input_dir, "target", celltype_target)

cat("Train:", ncol(sce_reference), "cells x", nrow(sce_reference), "genes\n")
cat("Test: ", ncol(sce_target),  "cells x", nrow(sce_target),  "genes\n")

# ── Log-normalize from raw counts ─────────────────────────────────────
# SingleR expects log-normalized values (normalize_total + log1p),
# NOT the z-scored "normalized" layer from the Python pipeline.
# At high dropout some cells have zero total counts -> librarySizeFactors
# returns 0 and normalizeCounts aborts with "size factors should be
# positive". Floor size factors at a tiny positive value so those cells
# are kept (they will log-normalize to ~0 and SingleR still emits a
# prediction for them, matching test-set ordering downstream).
floor_sf <- function(sce) {
  sf <- librarySizeFactors(sce)
  n_zero <- sum(sf <= 0)
  if (n_zero > 0) {
    cat("  flooring", n_zero, "of", length(sf),
        "non-positive size factors to 1e-8\n")
  }
  sizeFactors(sce) <- pmax(sf, 1e-8)
  sce
}
sce_reference <- logNormCounts(floor_sf(sce_reference))
sce_target  <- logNormCounts(floor_sf(sce_target))

# ── Extract training labels ────────────────────────────────────────────
reference_labels <- colData(sce_reference)[[celltype_reference]]
target_labels  <- colData(sce_target)[[celltype_target]]

cat("Training label distribution:\n")
print(table(reference_labels))
cat("\nTest label distribution:\n")
print(table(target_labels))

# ── Load inverse_dict_reference from CSV ──────────────────────────────────
inv_csv <- read.csv(file.path(input_dir, "inverse_dict_reference.csv"),
                    header = FALSE, stringsAsFactors = FALSE)
label_to_int <- setNames(as.integer(inv_csv$V1), inv_csv$V2)

# ── Create output directory ────────────────────────────────────────────
singler_dir <- file.path(savedir, "SingleR")
dir.create(singler_dir, showWarnings = FALSE, recursive = TRUE)

# ── Run SingleR (deterministic — run once, copy to all seeds) ─────────
cat("\nRunning SingleR ...\n")
t_start <- proc.time()

# num.threads is opt-in via SINGLER_THREADS (default 1 preserves prior behaviour).
# Large datasets (e.g. merfish_wholebrain_ft) time out single-threaded, so their
# sbatch sets SINGLER_THREADS=<cpus> to parallelise the classification.
singler_results <- SingleR(
  test   = sce_target,
  ref    = sce_reference,
  labels = reference_labels,
  num.threads = as.integer(Sys.getenv("SINGLER_THREADS", unset = "1"))
)

t_elapsed <- (proc.time() - t_start)["elapsed"]
cat("SingleR completed in", round(t_elapsed, 1), "seconds\n")

# Extract predictions
pred_labels <- singler_results$labels
pred_int <- ifelse(pred_labels %in% names(label_to_int),
                   label_to_int[pred_labels],
                   -1L)
pred_int <- as.integer(pred_int)

n_unknown <- sum(pred_int == -1L)
if (n_unknown > 0) {
  cat("WARNING:", n_unknown, "predictions mapped to unknown label (-1)\n")
}

# Build results dataframe
results_df <- data.frame(
  cell         = colnames(sce_target),
  predicted    = pred_labels,
  pred_int     = pred_int,
  ground_truth = target_labels,
  stringsAsFactors = FALSE
)
scores <- as.data.frame(singler_results$scores)
results_df <- cbind(results_df, scores)

# Save once (SingleR is deterministic — verified identical across seeds)
write.csv(results_df, file.path(singler_dir, "singler_predictions.csv"),
          row.names = FALSE)
saveRDS(singler_results, file.path(singler_dir, "singler_results.rds"))

# ── Save stats JSON ───────────────────────────────────────────────────
mem_info <- gc(verbose = FALSE)
peak_rss_kb <- sum(mem_info[, "max used"]) * 1024  # rough estimate in KB

stats <- list(
  runtime_sec = unname(t_elapsed),
  peak_rss_kb = peak_rss_kb,
  rounds      = rounds,
  n_reference     = ncol(sce_reference),
  n_target      = ncol(sce_target),
  n_genes     = nrow(sce_target),
  n_classes   = length(unique(reference_labels))
)
write_json(stats, file.path(singler_dir, "SingleR_stats.json"),
           pretty = TRUE, auto_unbox = TRUE)

cat("\nResults saved to", singler_dir, "\n")
cat("Run 'python singler_io.py convert' to generate .npy files.\n")
cat("Done.\n")
