#!/usr/bin/env python
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from westminster.gtex import match_tissue_targets, sqtl_keywords

"""
westminster_sqtl_gtex.py

Evaluate variant effect prediction accuracy on GTEx sQTL classification task,
matching model targets to GTEx tissues for per-tissue AUROC/AUPRC.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gtex_vcf_dir",
        default="/home/drk/seqnn/data/qtl_cat/sqtl_pip90",
        help="sQTL VCF directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gtex_out",
        help="Output directory for tissue metrics",
    )
    parser.add_argument(
        "-s",
        "--snp_stat",
        default="covgene/logSUM",
        help="SNP statistic. [Default: %(default)s]",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("sqtl_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_rows = []

    for tissue, keyword in sqtl_keywords.items():
        if args.verbose:
            print(tissue)

        pos_scores_file = f"{args.sqtl_dir}/{tissue}_pos/scores.h5"
        if not os.path.isfile(pos_scores_file):
            continue

        try:
            _match_tissue_targets(pos_scores_file, keyword, args.snp_stat, args.verbose)
        except ValueError:
            print(f"Skipping {tissue}: no matching targets.", file=sys.stderr)
            continue

        psnp_scores, _ = read_snp_scores(pos_scores_file, keyword, args.snp_stat)

        neg_scores_file = pos_scores_file.replace("_pos/", "_neg/")
        nsnp_scores, _ = read_snp_scores(neg_scores_file, keyword, args.snp_stat)

        Xp = list(psnp_scores.values())
        Xn = list(nsnp_scores.values())
        labels = [1] * len(Xp) + [0] * len(Xn)
        scores = Xp + Xn

        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        metrics_rows.append({"tissue": tissue, "auroc": auroc, "auprc": auprc})

        if args.verbose:
            print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    # form metrics table
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        f"{args.out_dir}/metrics.tsv", sep="\t", index=False, float_format="%.4f"
    )

    print("AUROC: %.4f" % np.mean(metrics_df.auroc))
    print("AUPRC: %.4f" % np.mean(metrics_df.auprc))


def _match_tissue_targets(
    scores_file: str,
    keyword: str,
    score_key: str = "covgene/logSUM",
    verbose: bool = False,
):
    """Read targets file and match tissue targets."""
    if score_key.startswith("gene/"):
        targets_name = "targets_gene.txt"
        gene_targets = True
    else:
        targets_name = "targets_cov.txt"
        gene_targets = False
    targets_file = scores_file.replace("scores.h5", targets_name)
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    match_tis = match_tissue_targets(targets_df, keyword, gene_targets, verbose)

    if len(match_tis) == 0:
        raise ValueError(f"WARNING: No targets matched keyword '{keyword}'.")
    return match_tis


def _load_hdf5_data(scores_file: str, score_key: str):
    """Load basic HDF5 datasets and build index mappings."""
    with h5py.File(scores_file, "r") as h5_file:
        snps = [s.decode("utf-8") for s in h5_file["snp"][:]]
        gene_ids_full = [g.decode("utf-8") for g in h5_file["gene_ids"][:]]

        snp_idx_arr = h5_file["snp_idx"][:]
        gene_idx_arr = h5_file["gene_idx"][:]

        if score_key not in h5_file:
            raise KeyError(f"Score key '{score_key}' not found in {scores_file}")
        stat_matrix = h5_file[score_key][:]  # (M, T)

    # Build index mappings
    snp_to_index = {snp: i for i, snp in enumerate(snps)}

    # Build mapping (snp_idx, gene_idx) -> row
    pair_row = {}
    for row_i, (si, gi) in enumerate(zip(snp_idx_arr, gene_idx_arr)):
        pair_row[(int(si), int(gi))] = row_i

    return {
        "snps": snps,
        "gene_ids_full": gene_ids_full,
        "stat_matrix": stat_matrix,
        "snp_to_index": snp_to_index,
        "pair_row": pair_row,
    }


def _aggregate_scores_across_genes(data: dict, match_tis: np.ndarray):
    """Aggregate absolute scores across all genes for each SNP."""
    snp_scores = {snp: 0.0 for snp in data["snps"]}
    scored_snps = set()

    for (si, gi), row in data["pair_row"].items():
        snp = data["snps"][si]
        scored_snps.add(snp)
        vals = data["stat_matrix"][row, match_tis].astype("float32")
        score = float(np.mean(vals))
        snp_scores[snp] += np.abs(score)

    return snp_scores, scored_snps


def read_snp_scores(scores_file, keyword, score_key="covgene/logSUM"):
    """Return per-SNP aggregated absolute scores from gene-level HDF5."""
    match_tis = _match_tissue_targets(scores_file, keyword, score_key)
    data = _load_hdf5_data(scores_file, score_key)
    return _aggregate_scores_across_genes(data, match_tis)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
