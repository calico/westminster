#!/usr/bin/env python
import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import pybedtools
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from westminster.gtex import (
    match_tissue_targets,
    tissue_keywords,
    trim_dot,
    vcf_tss_dist,
)

"""
westminster_gtexg_coef.py

Evaluate concordance of variant effect prediction sign classifcation
and coefficient correlations.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--genes_bed_file",
        default=f"{os.environ['HG38']}/genes/gencode48/gencode48_basic_protein_tss2.bed",
        help="BED file of gene TSS positions, for computing variant distance to TSS",
    )
    parser.add_argument(
        "-g",
        "--gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_fine/susie_pip90r",
        help="GTEx VCF directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="coef_out",
        help="Output directory for tissue metrics",
    )
    parser.add_argument(
        "-s",
        "--snp_stat",
        default="logSUM",
        help="SNP statistic. [Default: %(default)s]",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("gtex_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_rows = []

    for tissue, keyword in tissue_keywords.items():
        if args.verbose:
            print(tissue)

        # read causal variants
        eqtl_df = read_eqtl(tissue, args.gtex_vcf_dir)
        if eqtl_df is not None:
            # read model predictions
            gtex_scores_file = f"{args.gtex_dir}/{tissue}_pos/scores.h5"
            try:
                eqtl_df = add_scores(
                    gtex_scores_file,
                    keyword,
                    eqtl_df,
                    args.snp_stat,
                    verbose=args.verbose,
                )
            except ValueError:
                print(f"Skipping {tissue} due to missing targets.", file=sys.stderr)
                continue

            # sign AUROC/AUPRC
            sign_auroc = roc_auc_score(eqtl_df.coef > 0, eqtl_df.score)
            sign_auprc = average_precision_score(eqtl_df.coef > 0, eqtl_df.score)

            # SpearmanR
            coef_r = spearmanr(eqtl_df.coef, eqtl_df.score)[0]

            # measured metrics
            meas_frac = np.mean(eqtl_df.measured)
            eqtl_meas_df = eqtl_df[eqtl_df.measured]
            sign_auroc_meas = roc_auc_score(eqtl_meas_df.coef > 0, eqtl_meas_df.score)
            sign_auprc_meas = average_precision_score(
                eqtl_meas_df.coef > 0, eqtl_meas_df.score
            )
            coef_r_meas = spearmanr(eqtl_meas_df.coef, eqtl_meas_df.score)[0]

            # read pos+neg per-SNP aggregated scores
            psnp_scores, _ = read_snp_scores(gtex_scores_file, keyword, args.snp_stat)
            gtex_nscores_file = gtex_scores_file.replace("_pos", "_neg")
            nsnp_scores, neg_scored_snps = read_snp_scores(
                gtex_nscores_file, keyword, args.snp_stat
            )

            # classification AUROC
            Xp = list(psnp_scores.values())
            Xn = list(nsnp_scores.values())
            class_labels = [1] * len(Xp) + [0] * len(Xn)
            class_scores = Xp + Xn
            class_auroc = roc_auc_score(class_labels, class_scores)
            class_auprc = average_precision_score(class_labels, class_scores)

            # measured classification AUROC
            measured_snps = set(eqtl_meas_df.variant)
            Xp_m = [v for s, v in psnp_scores.items() if s in measured_snps]
            Xn_m = [v for s, v in nsnp_scores.items() if s in neg_scored_snps]
            class_labels_meas = [1] * len(Xp_m) + [0] * len(Xn_m)
            class_scores_meas = Xp_m + Xn_m
            class_auroc_meas = roc_auc_score(class_labels_meas, class_scores_meas)
            class_auprc_meas = average_precision_score(
                class_labels_meas, class_scores_meas
            )

            # compute TSS distances (dict-based: eqtl_df order != VCF order)
            tissue_vcf_file = f"{args.gtex_vcf_dir}/{tissue}_pos.vcf"
            pos_tss_arr = vcf_tss_dist(tissue_vcf_file, args.genes_bed_file)
            pos_vcf_variants = [
                line.split()[2]
                for line in open(tissue_vcf_file)
                if not line.startswith("##")
            ]
            pos_tss_map = dict(zip(pos_vcf_variants, pos_tss_arr))

            neg_vcf_file = f"{args.gtex_vcf_dir}/{tissue}_neg.vcf"
            neg_tss_arr = vcf_tss_dist(neg_vcf_file, args.genes_bed_file)
            neg_vcf_variants = [
                line.split()[2]
                for line in open(neg_vcf_file)
                if not line.startswith("##")
            ]
            neg_tss_map = dict(zip(neg_vcf_variants, neg_tss_arr))

            # write combined variant table
            pos_var_df = (
                eqtl_df.groupby("variant", sort=False)
                .agg(coef=("coef", "mean"))
                .reset_index()
            )
            pos_var_df = pos_var_df[pos_var_df.variant.isin(pos_tss_map)]
            pos_var_df["pred"] = [psnp_scores.get(v, 0.0) for v in pos_var_df.variant]
            pos_var_df["tss_dist"] = [pos_tss_map[v] for v in pos_var_df.variant]
            pos_var_df["label"] = "pos"

            neg_variants = [v for v in nsnp_scores if v in neg_tss_map]
            neg_var_df = pd.DataFrame(
                {
                    "variant": neg_variants,
                    "label": "neg",
                    "coef": np.nan,
                    "pred": [nsnp_scores[v] for v in neg_variants],
                    "tss_dist": [neg_tss_map[v] for v in neg_variants],
                }
            )

            scatter_df = pd.concat(
                [
                    pos_var_df[["variant", "label", "coef", "pred", "tss_dist"]],
                    neg_var_df,
                ],
                ignore_index=True,
            )
            scatter_df.to_csv(f"{args.out_dir}/{tissue}.tsv", index=False, sep="\t")

            # save metrics
            metrics_rows.append(
                {
                    "tissue": tissue,
                    "auroc_sign": sign_auroc,
                    "auprc_sign": sign_auprc,
                    "spearmanr": coef_r,
                    "auroc_class": class_auroc,
                    "auprc_class": class_auprc,
                    "measured": meas_frac,
                    "measured_auroc_sign": sign_auroc_meas,
                    "measured_auprc_sign": sign_auprc_meas,
                    "measured_spearmanr": coef_r_meas,
                    "measured_auroc_class": class_auroc_meas,
                    "measured_auprc_class": class_auprc_meas,
                }
            )

            if args.verbose:
                print("")

    pybedtools.cleanup()

    # form metrics table
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        f"{args.out_dir}/metrics.tsv", sep="\t", index=False, float_format="%.4f"
    )

    # summarize
    print("Sign AUROC:  %.4f" % np.mean(metrics_df.auroc_sign))
    print("Sign AUPRC:  %.4f" % np.mean(metrics_df.auprc_sign))
    print("SpearmanR:   %.4f" % np.mean(metrics_df.spearmanr))
    print("Class AUROC: %.4f" % np.mean(metrics_df.auroc_class))
    print("Class AUPRC: %.4f" % np.mean(metrics_df.auprc_class))
    print("Measured fraction: %.4f" % np.mean(metrics_df.measured))
    print("Measured Sign AUROC:  %.4f" % np.mean(metrics_df.measured_auroc_sign))
    print("Measured Sign AUPRC:  %.4f" % np.mean(metrics_df.measured_auprc_sign))
    print("Measured SpearmanR:   %.4f" % np.mean(metrics_df.measured_spearmanr))
    print("Measured Class AUROC: %.4f" % np.mean(metrics_df.measured_auroc_class))
    print("Measured Class AUPRC: %.4f" % np.mean(metrics_df.measured_auprc_class))


def read_eqtl(tissue: str, gtex_vcf_dir: str, pip_t: float = 0.9):
    """Reads eQTLs from SUSIE output.

    Args:
      tissue (str): Tissue name.
      gtex_vcf_dir (str): GTEx VCF directory.
      pip_t (float): PIP threshold.

    Returns:
      eqtl_df (pd.DataFrame): eQTL dataframe, or None if tissue skipped.
    """
    susie_dir = "/home/drk/seqnn/data/gtex_fine/tissues_susie"

    # read causal variants
    eqtl_file = f"{susie_dir}/{tissue}.tsv"
    df_eqtl = pd.read_csv(eqtl_file, sep="\t", index_col=0)

    # pip filter
    pip_match = re.search(r"_pip(\d+).+$", gtex_vcf_dir).group(1)
    pip_t = float(pip_match) / 100
    assert pip_t > 0 and pip_t <= 1
    df_causal = df_eqtl[df_eqtl.pip > pip_t]

    # make table
    tissue_vcf_file = f"{gtex_vcf_dir}/{tissue}_pos.vcf"
    if not os.path.isfile(tissue_vcf_file):
        eqtl_df = None
    else:
        # create dataframe
        eqtl_df = pd.DataFrame(
            {
                "variant": df_causal.variant,
                "gene": [trim_dot(gene_id) for gene_id in df_causal.gene],
                "coef": df_causal.beta_posterior,
                "allele1": df_causal.allele1,
            }
        )
    return eqtl_df


def _match_tissue_targets(
    gtex_scores_file: str,
    keyword: str,
    score_key: str = "logSUM",
    verbose: bool = False,
):
    """Read targets file and match tissue targets."""
    if score_key.startswith("gene/"):
        targets_name = "targets_gene.txt"
        gene_targets = True
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
    """Load basic HDF5 datasets and build index mappings.

    Args:
        gtex_scores_file: Path to the HDF5 scores file
        score_key: Score key to read from HDF5

    Returns:
        Dictionary containing loaded data and mappings
    """
    with h5py.File(gtex_scores_file, "r") as h5_file:
        # Basic datasets
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
    eqtl_df: pd.DataFrame,
    score_key: str = "logSUM",
    verbose: bool = False,
):
    """
    Read from gene-specific HDF5.
      1. Determine target indices matching the tissue keyword.
      2. Build (snp_idx, gene_idx) -> row map.
      3. For each eQTL (variant, gene) fetch its row, average selected target columns, flip sign if needed.

    Args:
      gtex_scores_file (str): Path to the gene-specific HDF5 file.
      keyword (str): Tissue keyword to match.
      eqtl_df (pd.DataFrame): DataFrame containing eQTL information.
      score_key (str): Key for the score to retrieve from HDF5.
      verbose (bool): If True, print verbose output.

    Returns:
      pd.DataFrame: Updated eQTL DataFrame with scores.
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
    for _, eqtl in eqtl_df.iterrows():
        variant = eqtl.variant
        gene_trim = eqtl.gene  # already trimmed upstream
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
            if ref_a != eqtl.allele1:
                sgs *= -1
        scores_out.append(sgs)
        measured.append(meas)

    # attach score
    eqtl_df["score"] = scores_out
    eqtl_df["measured"] = measured
    return eqtl_df


def _aggregate_scores_across_genes(data: dict, match_tis: np.ndarray):
    """Aggregate absolute scores across all genes for each SNP.

    Args:
        data: Data dictionary from _load_hdf5_data()
        match_tis: Array of target indices that match the tissue

    Returns:
        snp_scores: Dictionary mapping SNP -> aggregated score
        scored_snps: Set of SNPs with at least one gene pair
    """
    snp_scores = {snp: 0.0 for snp in data["snps"]}
    scored_snps = set()

    for (si, gi), row in data["pair_row"].items():
        snp = data["snps"][si]
        scored_snps.add(snp)
        vals = data["stat_matrix"][row, match_tis].astype("float32")
        score = float(np.mean(vals))
        snp_scores[snp] += np.abs(score)

    return snp_scores, scored_snps


def read_snp_scores(gtex_scores_file, keyword, score_key="logSUM"):
    """Return per-SNP aggregated absolute scores from gene-level HDF5.

    Args:
      gtex_scores_file (str): Path to HDF5 scores file.
      keyword (str): Tissue keyword for matching GTEx targets.
      score_key (str): Score key in HDF5 file.

    Returns:
      snp_scores (dict): SNP -> aggregated abs score.
      scored_snps (set): SNPs with at least one gene pair.
    """
    match_tis = _match_tissue_targets(gtex_scores_file, keyword, score_key)
    data = _load_hdf5_data(gtex_scores_file, score_key)
    return _aggregate_scores_across_genes(data, match_tis)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
