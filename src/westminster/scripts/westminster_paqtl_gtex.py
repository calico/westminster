#!/usr/bin/env python
import argparse
import os
import sys

import glob
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from westminster.gtex import (
    match_tissue_targets,
    txrev_keywords,
    gtexv11_keywords,
    trim_dot,
)

"""
westminster_paqtl_gtex.py

Evaluate variant effect prediction accuracy on GTEx paQTL classification task,
matching model targets to GTEx tissues for per-tissue AUROC/AUPRC.

Each variant is scored at exactly one designated gene (INFO GENE=): positives
at their paQTL gene, negatives at the matched-negative assigned gene. This
avoids the gene-density bias of summing |score| across every cis gene.
"""

# distance INFO tag and per-tissue-table column for paQTL
DIST_TAG = "PD"
DIST_COL = "pas_dist"


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gtex_out",
        help="Output directory for tissue metrics",
    )
    parser.add_argument(
        "-g",
        "--gtex_vcf_dir",
        required=True,
        help="GTEx paQTL VCF directory with {tissue}_pos.vcf / {tissue}_neg.vcf "
        "(INFO carries GENE and PD)",
    )
    parser.add_argument(
        "-s",
        "--snp_stat",
        default="covgene/PA",
        help="SNP statistic. [Default: %(default)s]",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("paqtl_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    keyword_lookup = {
        t.replace("GTEx_txrev_", ""): kw for t, kw in txrev_keywords.items()
    }
    keyword_lookup.update(gtexv11_keywords)

    metrics_rows = []

    for pos_dir in sorted(glob.glob(f"{args.paqtl_dir}/*_pos")):
        tissue_label = os.path.basename(pos_dir).removesuffix("_pos")
        if tissue_label == "merge":
            continue  # AlphaGenome's merge_pos staging dir, not a tissue
        if tissue_label not in keyword_lookup:
            print(f"Skipping {tissue_label}: no keyword mapping.", file=sys.stderr)
            continue
        keyword = keyword_lookup[tissue_label]

        if args.verbose:
            print(tissue_label)

        pos_scores_file = f"{pos_dir}/scores.h5"
        neg_scores_file = pos_scores_file.replace("_pos/", "_neg/")
        if not os.path.isfile(pos_scores_file) or not os.path.isfile(neg_scores_file):
            continue

        # read designated genes from the per-tissue VCFs
        pos_df = read_qtl_vcf(f"{args.gtex_vcf_dir}/{tissue_label}_pos.vcf")
        neg_df = read_qtl_vcf(f"{args.gtex_vcf_dir}/{tissue_label}_neg.vcf")
        if pos_df is None or neg_df is None:
            print(f"Skipping {tissue_label}: missing VCF.", file=sys.stderr)
            continue

        # score each variant at its single designated gene
        try:
            pos_df = add_scores(
                pos_scores_file, keyword, pos_df, args.snp_stat, verbose=args.verbose
            )
            neg_df = add_scores(
                neg_scores_file, keyword, neg_df, args.snp_stat, verbose=args.verbose
            )
        except ValueError:
            print(f"Skipping {tissue_label}: no matching targets.", file=sys.stderr)
            continue

        # classification AUROC/AUPRC: |designated-gene score|, positives vs
        # matched negatives
        Xp = np.abs(pos_df.score.values)
        Xn = np.abs(neg_df.score.values)
        labels = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        scores = np.r_[Xp, Xn]
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        # measured classification: restrict each side to variants whose
        # designated (variant, gene) pair was scored in-window
        meas_frac = float(
            np.mean(np.r_[pos_df.measured.values, neg_df.measured.values])
        )
        Xp_m = np.abs(pos_df.score.values[pos_df.measured.values])
        Xn_m = np.abs(neg_df.score.values[neg_df.measured.values])
        labels_m = np.r_[np.ones(len(Xp_m)), np.zeros(len(Xn_m))]
        scores_m = np.r_[Xp_m, Xn_m]
        auroc_meas = roc_auc_score(labels_m, scores_m)
        auprc_meas = average_precision_score(labels_m, scores_m)

        metrics_rows.append(
            {
                "tissue": tissue_label,
                "auroc": auroc,
                "auprc": auprc,
                "measured": meas_frac,
                "measured_auroc": auroc_meas,
                "measured_auprc": auprc_meas,
            }
        )

        # write per-variant table: one designated-gene score per variant
        pos_var = pos_df.rename(columns={"score": "pred"}).assign(label="pos")
        neg_var = neg_df.rename(columns={"score": "pred"}).assign(label="neg")
        scatter_cols = ["variant", "label", "pred", "measured", DIST_COL]
        scatter_df = pd.concat(
            [pos_var[scatter_cols], neg_var[scatter_cols]], ignore_index=True
        )
        scatter_df.to_csv(f"{args.out_dir}/{tissue_label}.tsv", index=False, sep="\t")

        if args.verbose:
            print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    # form metrics table
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        f"{args.out_dir}/metrics.tsv", sep="\t", index=False, float_format="%.4f"
    )

    print("AUROC:             %.4f" % np.mean(metrics_df.auroc))
    print("AUPRC:             %.4f" % np.mean(metrics_df.auprc))
    print("Measured fraction: %.4f" % np.mean(metrics_df.measured))
    print("Measured AUROC:    %.4f" % np.mean(metrics_df.measured_auroc))
    print("Measured AUPRC:    %.4f" % np.mean(metrics_df.measured_auprc))


def read_qtl_vcf(vcf_file: str):
    """Read variants from a per-tissue paQTL VCF (INFO carries GENE and PD).

    Works for both {tissue}_pos.vcf and {tissue}_neg.vcf — both carry one
    designated gene per variant; positive-only tags (SLOPE, NLP) are ignored.
    Returns one row per variant with the columns add_scores() needs.
    """
    if not os.path.isfile(vcf_file):
        return None
    rows = []
    for line in open(vcf_file):
        if line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        vid = cols[2]
        ref = cols[3]
        info = dict(f.split("=", 1) for f in cols[7].split(";") if "=" in f)
        gene = trim_dot(info.get("GENE", ""))
        if not gene:
            continue
        dist_raw = info.get(DIST_TAG, ".")
        dist = np.nan if dist_raw == "." else float(dist_raw)
        rows.append({"variant": vid, "gene": gene, "allele1": ref, DIST_COL: dist})
    return pd.DataFrame(rows) if rows else None


def _match_tissue_targets(
    gtex_scores_file: str,
    keyword: str,
    score_key: str = "covgene/PA",
    verbose: bool = False,
):
    """Read targets file and match tissue targets."""
    if score_key.startswith("gene/"):
        targets_name = "targets_gene.txt"
        gene_targets = True
    elif score_key.startswith("covgene/"):
        # covgene/ stats span the gene-track subset, indexed by targets_covgene.txt
        targets_name = "targets_covgene.txt"
        gene_targets = False
    else:
        targets_name = "targets_cov.txt"
        gene_targets = False
    targets_file = gtex_scores_file.replace("scores.h5", targets_name)
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    match_tis = match_tissue_targets(targets_df, keyword, gene_targets, verbose)

    if len(match_tis) == 0:
        raise ValueError(f"WARNING: No targets matched keyword '{keyword}'.")
    return match_tis


def _load_hdf5_data(gtex_scores_file: str, score_key: str):
    """Load basic HDF5 datasets and build index mappings."""
    with h5py.File(gtex_scores_file, "r") as h5_file:
        snps = [s.decode("utf-8") for s in h5_file["snp"][:]]
        gene_ids_full = [g.decode("utf-8") for g in h5_file["gene_ids"][:]]

        snp_idx_arr = h5_file["snp_idx"][:]
        gene_idx_arr = h5_file["gene_idx"][:]

        if score_key not in h5_file:
            raise KeyError(f"Score key '{score_key}' not found in {gtex_scores_file}")
        stat_matrix = h5_file[score_key][:]  # (M, T)

        # Load reference alleles if available (for allele flipping)
        ref_alleles = None
        if "ref_allele" in h5_file:
            ref_alleles = [a.decode("utf-8") for a in h5_file["ref_allele"][:]]

    # Build index mappings
    snp_to_index = {snp: i for i, snp in enumerate(snps)}

    # Gene ID mapping (handle versioned IDs)
    gene_ids_trimmed = [trim_dot(g) for g in gene_ids_full]
    trimmed_to_index = {}
    for idx, gtrim in enumerate(gene_ids_trimmed):
        if gtrim not in trimmed_to_index:  # keep first occurrence
            trimmed_to_index[gtrim] = idx

    # Build mapping (snp_idx, gene_idx) -> row
    pair_row = {}
    for row_i, (si, gi) in enumerate(zip(snp_idx_arr, gene_idx_arr)):
        pair_row[(int(si), int(gi))] = row_i

    return {
        "snps": snps,
        "gene_ids_full": gene_ids_full,
        "ref_alleles": ref_alleles,
        "stat_matrix": stat_matrix,
        "snp_to_index": snp_to_index,
        "trimmed_to_index": trimmed_to_index,
        "pair_row": pair_row,
    }


def add_scores(
    gtex_scores_file: str,
    keyword: str,
    qtl_df: pd.DataFrame,
    score_key: str = "covgene/PA",
    verbose: bool = False,
):
    """Score each (variant, gene) pair at its designated gene.

    1. Determine target indices matching the tissue keyword.
    2. Build (snp_idx, gene_idx) -> row map.
    3. For each (variant, gene) fetch its row, average the matched target
       columns, flip sign if the VCF allele1 differs from the HDF5 reference.

    Returns the input DataFrame with `score` and `measured` columns added;
    `measured` is False (score 0.0) when the pair is not present in the HDF5
    (variant or gene out of the model's predictable window).
    """
    # Match tissue targets
    match_tis = _match_tissue_targets(gtex_scores_file, keyword, score_key, verbose)

    # Load HDF5 data and mappings
    data = _load_hdf5_data(gtex_scores_file, score_key)

    # Reference allele map for flipping
    snp_ref = (
        dict(zip(data["snps"], data["ref_alleles"])) if data["ref_alleles"] else {}
    )

    scores_out = []
    measured = []
    for _, qtl in qtl_df.iterrows():
        variant = qtl.variant
        gene_trim = qtl.gene  # already trimmed upstream
        si = data["snp_to_index"].get(variant, None)
        gi = data["trimmed_to_index"].get(gene_trim, None)
        if si is not None and gi is not None:
            row = data["pair_row"].get((si, gi), None)
            if row is not None:
                vals = data["stat_matrix"][row, match_tis].astype("float32")
                sgs = float(np.mean(vals))
                meas = True
            else:
                sgs = 0.0
                meas = False
        else:
            sgs = 0.0
            meas = False

        # flip sign if allele1 != reference
        if sgs != 0.0 and si is not None and data["ref_alleles"]:
            ref_a = snp_ref[variant]
            if ref_a != qtl.allele1:
                sgs *= -1
        scores_out.append(sgs)
        measured.append(meas)

    qtl_df["score"] = scores_out
    qtl_df["measured"] = measured
    return qtl_df


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
