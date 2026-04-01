#!/usr/bin/env python
import argparse
import os
import pdb
import re
import sys

import h5py
import numpy as np
import pandas as pd
import pybedtools
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from westminster.gtex import match_tissue_targets, tissue_keywords, vcf_tss_dist

"""
westminster_gtex_coef.py

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
        "-m",
        "--min_variants",
        default=32,
        type=int,
        help="Minimum number of variants for tissue to be included",
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
        tissue_vcf_file = f"{args.gtex_vcf_dir}/{tissue}_pos.vcf"
        eqtl_df = read_eqtl(tissue, tissue_vcf_file)
        if eqtl_df is not None and eqtl_df.shape[0] > args.min_variants:
            # read model predictions
            gtex_scores_file = f"{args.gtex_dir}/{tissue}_pos/scores.h5"
            try:
                variant_scores = read_scores(
                    gtex_scores_file,
                    keyword,
                    eqtl_df,
                    args.snp_stat,
                    verbose=args.verbose,
                )
            except TypeError:
                print(f"Tracks matching {tissue} are missing", file=sys.stderr)
                continue

            # compute variant-TSS distances
            eqtl_df["tss_dist"] = vcf_tss_dist(tissue_vcf_file, args.genes_bed_file)

            # compute sign AUROCs
            variant_sign = eqtl_df[eqtl_df.consistent].sign
            cvariant_scores = variant_scores[eqtl_df.consistent]
            sign_auroc = roc_auc_score(variant_sign, cvariant_scores)
            sign_auprc = average_precision_score(variant_sign, cvariant_scores)

            # compute SpearmanR
            variant_coef = eqtl_df[eqtl_df.consistent].coef
            coef_r = spearmanr(variant_coef, cvariant_scores)[0]

            # read negative (control) variant scores and distances
            neg_scores = read_neg_scores(gtex_scores_file, keyword, args.snp_stat)
            neg_vcf_file = f"{args.gtex_vcf_dir}/{tissue}_neg.vcf"
            neg_variants = np.array(
                [
                    line.split()[2]
                    for line in open(neg_vcf_file)
                    if not line.startswith("#")
                ]
            )
            neg_tss_dist = vcf_tss_dist(neg_vcf_file, args.genes_bed_file)

            # classification AUROC
            pos_abs = np.abs(variant_scores)
            neg_abs = np.abs(neg_scores)
            class_labels = np.concatenate(
                [np.ones(len(pos_abs)), np.zeros(len(neg_abs))]
            )
            class_scores = np.concatenate([pos_abs, neg_abs])
            class_auroc = roc_auc_score(class_labels, class_scores)
            class_auprc = average_precision_score(class_labels, class_scores)

            # write combined variant table
            pos_df = pd.DataFrame(
                {
                    "variant": eqtl_df.variant,
                    "label": "pos",
                    "coef": eqtl_df.coef,
                    "pred": variant_scores,
                    "tss_dist": eqtl_df.tss_dist,
                }
            )
            neg_df = pd.DataFrame(
                {
                    "variant": neg_variants,
                    "label": "neg",
                    "coef": np.nan,
                    "pred": neg_abs,
                    "tss_dist": neg_tss_dist,
                }
            )
            scatter_df = pd.concat([pos_df, neg_df], ignore_index=True)
            scatter_df.to_csv(f"{args.out_dir}/{tissue}.tsv", index=False, sep="\t")

            # save metrics
            metrics_rows.append(
                {
                    "tissue": tissue,
                    "variants": eqtl_df.shape[0],
                    "auroc_sign": sign_auroc,
                    "auprc_sign": sign_auprc,
                    "spearmanr": coef_r,
                    "auroc_class": class_auroc,
                    "auprc_class": class_auprc,
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


def read_eqtl(tissue: str, tissue_vcf_file: str, pip_t: float = 0.9):
    """Reads eQTLs from SUSIE output.

    Args:
      tissue (str): Tissue name.
      tissue_vcf_file (str): Path to tissue VCF file.
      pip_t (float): PIP threshold.

    Returns:
      eqtl_df (pd.DataFrame): eQTL dataframe, or None if tissue skipped.
    """
    susie_dir = "/home/drk/seqnn/data/gtex_fine/tissues_susie"

    # read causal variants
    eqtl_file = f"{susie_dir}/{tissue}.tsv"
    df_eqtl = pd.read_csv(eqtl_file, sep="\t", index_col=0)

    # pip filter
    pip_match = re.search(r"_pip(\d+)", tissue_vcf_file).group(1)
    pip_t = float(pip_match) / 100
    assert pip_t > 0 and pip_t <= 1
    df_causal = df_eqtl[df_eqtl.pip > pip_t]

    # remove variants with inconsistent signs
    variant_a1 = {}
    variant_sign = {}
    variant_beta = {}
    variant_class = {}
    inconsistent_variants = set()
    for variant in df_causal.itertuples():
        vid = variant.variant
        vsign = variant.beta_posterior > 0

        # classify variant type
        if len(variant.allele1) == len(variant.allele2):
            variant_class[vid] = "SNP"
        elif len(variant.allele1) < len(variant.allele2):
            variant_class[vid] = "insertion"
        else:
            variant_class[vid] = "deletion"

        variant_a1[vid] = variant.allele1
        variant_beta.setdefault(vid, []).append(variant.beta_posterior)
        if vid in variant_sign:
            if variant_sign[vid] != vsign:
                inconsistent_variants.add(vid)
        else:
            variant_sign[vid] = vsign

    # average beta's across genes
    for vid in variant_beta:
        variant_beta[vid] = np.mean(variant_beta[vid])

    # order variants
    if not os.path.isfile(tissue_vcf_file):
        eqtl_df = None
    else:
        pred_variants = np.array(
            [
                line.split()[2]
                for line in open(tissue_vcf_file)
                if not line.startswith("#")
            ]
        )
        consistent_mask = np.array(
            [vid not in inconsistent_variants for vid in pred_variants]
        )

        # create dataframe
        eqtl_df = pd.DataFrame(
            {
                "variant": pred_variants,
                "coef": [variant_beta[vid] for vid in pred_variants],
                "sign": [variant_sign[vid] for vid in pred_variants],
                "allele": [variant_a1[vid] for vid in pred_variants],
                "consistent": consistent_mask,
                "class": [variant_class[vid] for vid in pred_variants],
            }
        )
    return eqtl_df


def read_scores(
    gtex_scores_file: str,
    keyword: str,
    eqtl_df: pd.DataFrame,
    score_key: str,
    verbose: bool = False,
):
    """Read eQTL RNA predictions for the given tissue.

    Args:
      gtex_scores_file (str): Variant scores HDF5.
      tissue_keyword (str): tissue keyword, for matching GTEx targets
      eqtl_df (pd.DataFrame): eQTL dataframe
      score_key (str): score key in HDF5 file
      verbose (bool): Print matching targets.

    Returns:
      np.array: eQTL predictions
    """
    targets_file = gtex_scores_file.replace("scores.h5", "targets_cov.txt")
    # TEMP handling original format
    try:
        targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)
    except FileNotFoundError:
        targets_file = targets_file.replace("targets_cov.txt", "targets.txt")
        targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    match_tis = match_tissue_targets(targets_df, keyword, verbose=verbose)

    with h5py.File(gtex_scores_file, "r") as gtex_scores_h5:
        score_ref = np.array(
            [ref.decode("UTF-8") for ref in gtex_scores_h5["ref_allele"]]
        )

        # mean across targets
        variant_scores = gtex_scores_h5[score_key][..., match_tis].mean(
            axis=-1, dtype="float32"
        )
        variant_scores = np.arcsinh(variant_scores)

    # strip prefix for sign logic (cov/logSUM -> logSUM)
    bare_key = score_key.split("/")[-1]
    if bare_key in ["SUM", "logSUM", "SAD", "logSAD"]:
        # flip signs
        sad_flip = score_ref != eqtl_df.allele
        variant_scores[sad_flip] = -variant_scores[sad_flip]
    else:
        # unsigned score
        variant_scores = np.abs(variant_scores)

    return variant_scores


def read_neg_scores(
    gtex_scores_file: str,
    keyword: str,
    score_key: str,
    verbose: bool = False,
):
    """Read negative (control) variant scores for the given tissue.

    Args:
      gtex_scores_file (str): Positive variant scores HDF5 (path rewritten to _neg).
      keyword (str): tissue keyword, for matching GTEx targets
      score_key (str): score key in HDF5 file
      verbose (bool): Print matching targets.

    Returns:
      np.array: Negative variant scores (arcsinh-transformed).
    """
    gtex_nscores_file = gtex_scores_file.replace("_pos", "_neg")

    targets_file = gtex_nscores_file.replace("scores.h5", "targets_cov.txt")
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    match_tis = match_tissue_targets(targets_df, keyword, verbose=verbose)

    with h5py.File(gtex_nscores_file, "r") as gtex_scores_h5:
        # mean across targets
        neg_scores = gtex_scores_h5[score_key][..., match_tis].mean(
            axis=-1, dtype="float32"
        )
        neg_scores = np.arcsinh(neg_scores)

    return neg_scores


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
