#!/usr/bin/env python
import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

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
        "-o",
        "--out_dir",
        default="coef_out",
        help="Output directory for tissue metrics",
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
        "-p", "--plot", action="store_true", help="Generate tissue prediction plots"
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

    tissue_keywords = {
        "Adipose_Subcutaneous": "adipose",
        "Adipose_Visceral_Omentum": "adipose",
        "Adrenal_Gland": "adrenal_gland",
        "Artery_Aorta": "heart",
        "Artery_Tibial": "heart",
        "Brain_Cerebellum": "brain",
        "Brain_Cortex": "brain",
        "Breast_Mammary_Tissue": "breast",
        "Colon_Sigmoid": "colon",
        "Colon_Transverse": "colon",
        "Esophagus_Mucosa": "esophagus",
        "Esophagus_Muscularis": "esophagus",
        "Liver": "liver",
        "Lung": "lung",
        "Muscle_Skeletal": "muscle",
        "Nerve_Tibial": "nerve",
        "Ovary": "ovary",
        "Pancreas": "pancreas",
        "Pituitary": "pituitary",
        "Prostate": "prostate",
        "Skin_Not_Sun_Exposed_Suprapubic": "skin",
        "Spleen": "spleen",
        "Stomach": "stomach",
        "Testis": "testis",
        "Thyroid": "thyroid",
        "Whole_Blood": "blood",
    }
    # 'Cells_Cultured_fibroblasts': 'fibroblast',

    metrics_tissue = []
    metrics_sauroc = []
    metrics_cauroc = []
    metrics_r = []
    for tissue, keyword in tissue_keywords.items():
        if args.verbose:
            print(tissue)

        # read causal variants
        eqtl_df = read_eqtl(tissue, args.gtex_vcf_dir)
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
                variant_scores = variant_scores[eqtl_df.consistent]
            except TypeError:
                print(f"Tracks matching {tissue} are missing", file=sys.stderr)
                continue

            # compute sign AUROCs
            variant_sign = eqtl_df[eqtl_df.consistent].sign
            sign_auroc = roc_auc_score(variant_sign, variant_scores)

            # compute SpearmanR
            variant_coef = eqtl_df[eqtl_df.consistent].coef
            coef_r = spearmanr(variant_coef, variant_scores)[0]

            # classification AUROC
            class_auroc = classify_auroc(
                gtex_scores_file, keyword, variant_scores, args.snp_stat
            )

            if args.plot:
                # write table
                scatter_df = pd.DataFrame(
                    {
                        "variant": eqtl_df[eqtl_df.consistent].variant,
                        "coef": variant_coef,
                        "pred": variant_scores,
                    }
                )
                scatter_df.to_csv(f"{args.out_dir}/{tissue}.tsv", index=False, sep="\t")

                # scatterplot
                plt.figure(figsize=(6, 6))
                sns.scatterplot(x=variant_coef, y=variant_scores, alpha=0.5, s=20)
                plt.gca().set_xlabel("eQTL coefficient")
                plt.gca().set_ylabel("Variant effect prediction")
                plt.savefig(f"{args.out_dir}/{tissue}.png", dpi=300)

            # save
            metrics_tissue.append(tissue)
            metrics_sauroc.append(sign_auroc)
            metrics_cauroc.append(class_auroc)
            metrics_r.append(coef_r)

            if args.verbose:
                print("")

    # save metrics
    metrics_df = pd.DataFrame(
        {
            "tissue": metrics_tissue,
            "auroc_sign": metrics_sauroc,
            "spearmanr": metrics_r,
            "auroc_class": metrics_cauroc,
        }
    )
    metrics_df.to_csv(
        f"{args.out_dir}/metrics.tsv", sep="\t", index=False, float_format="%.4f"
    )

    # summarize
    print("Sign AUROC:  %.4f" % np.mean(metrics_df.auroc_sign))
    print("SpearmanR:   %.4f" % np.mean(metrics_df.spearmanr))
    print("Class AUROC: %.4f" % np.mean(metrics_df.auroc_class))


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
    pip_match = re.search(r"_pip(\d+)", gtex_vcf_dir).group(1)
    pip_t = float(pip_match) / 100
    assert pip_t > 0 and pip_t <= 1
    df_causal = df_eqtl[df_eqtl.pip > pip_t]

    # remove variants with inconsistent signs
    variant_a1 = {}
    variant_sign = {}
    variant_beta = {}
    inconsistent_variants = set()
    for variant in df_causal.itertuples():
        vid = variant.variant
        vsign = variant.beta_posterior > 0

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
    tissue_vcf_file = f"{gtex_vcf_dir}/{tissue}_pos.vcf"
    if not os.path.isfile(tissue_vcf_file):
        eqtl_df = None
    else:
        pred_variants = np.array(
            [
                line.split()[2]
                for line in open(tissue_vcf_file)
                if not line.startswith("##")
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
            }
        )
    return eqtl_df


def read_scores(
    gtex_scores_file: str,
    keyword: str,
    eqtl_df: pd.DataFrame,
    score_key: str = "SAD",
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
    with h5py.File(gtex_scores_file, "r") as gtex_scores_h5:
        score_ref = np.array(
            [ref.decode("UTF-8") for ref in gtex_scores_h5["ref_allele"]]
        )

        # determine matching GTEx targets
        target_ids = np.array(
            [ref.decode("UTF-8") for ref in gtex_scores_h5["target_ids"]]
        )
        target_labels = np.array(
            [ref.decode("UTF-8") for ref in gtex_scores_h5["target_labels"]]
        )
        match_tis = []
        for ti in range(len(target_ids)):
            if (
                target_ids[ti].find("GTEX") != -1
                and target_labels[ti].find(keyword) != -1
            ):
                if not keyword == "blood" or target_labels[ti].find("vessel") == -1:
                    if verbose:
                        print(ti, target_ids[ti], target_labels[ti])
                    match_tis.append(ti)
        match_tis = np.array(match_tis)

        # mean across targets
        variant_scores = gtex_scores_h5[score_key][..., match_tis].mean(
            axis=-1, dtype="float32"
        )
        variant_scores = np.arcsinh(variant_scores)

    # flip signs
    sad_flip = score_ref != eqtl_df.allele
    variant_scores[sad_flip] = -variant_scores[sad_flip]

    return variant_scores


def classify_auroc(
    gtex_scores_file: str,
    keyword: str,
    pos_scores: np.array,
    score_key: str = "SAD",
    verbose: bool = False,
):
    """Read eQTL RNA predictions for the given tissue.

    Args:
      gtex_scores_file (str): Variant scores HDF5.
      tissue_keyword (str): tissue keyword, for matching GTEx targets
      pos_scores (np.array): eQTL predictions
      score_key (str): score key in HDF5 file
      verbose (bool): Print matching targets.

    Returns:
      np.array: eQTL predictions
    """
    gtex_nscores_file = gtex_scores_file.replace("_pos", "_neg")
    with h5py.File(gtex_nscores_file, "r") as gtex_scores_h5:
        # determine matching GTEx targets
        target_ids = np.array(
            [ref.decode("UTF-8") for ref in gtex_scores_h5["target_ids"]]
        )
        target_labels = np.array(
            [ref.decode("UTF-8") for ref in gtex_scores_h5["target_labels"]]
        )
        match_tis = []
        for ti in range(len(target_ids)):
            if (
                target_ids[ti].find("GTEX") != -1
                and target_labels[ti].find(keyword) != -1
            ):
                if not keyword == "blood" or target_labels[ti].find("vessel") == -1:
                    if verbose:
                        print(ti, target_ids[ti], target_labels[ti])
                    match_tis.append(ti)
        match_tis = np.array(match_tis)

        # mean across targets
        neg_scores = gtex_scores_h5[score_key][..., match_tis].mean(
            axis=-1, dtype="float32"
        )
        neg_scores = np.arcsinh(neg_scores)

    pos_scores = np.abs(pos_scores)
    neg_scores = np.abs(neg_scores)
    X = np.concatenate([pos_scores, neg_scores])
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    return roc_auc_score(y, X)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
