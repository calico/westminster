#!/usr/bin/env python
import argparse
import os
import pdb
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
        if eqtl_df is not None:
            # read model predictions
            gtex_scores_file = f"{args.gtex_dir}/{tissue}_pos/scores.h5"
            try:
                eqtl_df = add_scores(
                    gtex_scores_file, keyword, eqtl_df, args.snp_stat, verbose=args.verbose
                )
            except ValueError:
                print(f"Skipping {tissue} due to missing targets.", file=sys.stderr)
                continue
 
            # compute AUROCs
            sign_auroc = roc_auc_score(eqtl_df.coef > 0, eqtl_df.score)

            # compute SpearmanR
            coef_r = spearmanr(eqtl_df.coef, eqtl_df.score)[0]

            # classification AUROC
            class_auroc = classify_auroc(gtex_scores_file, keyword, args.snp_stat)

            if args.plot:
                eqtl_df.to_csv(f"{args.out_dir}/{tissue}.tsv", index=False, sep="\t")

                # scatterplot
                plt.figure(figsize=(6, 6))
                sns.scatterplot(x=eqtl_df.coef, y=eqtl_df.score, alpha=0.5, s=20)
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


def _match_tissue_targets(gtex_scores_file: str, keyword: str, verbose: bool = False):
    """Match tissue targets based on keyword.

    Args:
        gtex_scores_file: Path to the HDF5 scores file
        keyword: Tissue keyword to match
        verbose: If True, print matching targets

    Returns:
        Array of target indices that match the tissue keyword
    """
    targets_file = gtex_scores_file.replace("scores.h5", "targets.txt")
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)
    target_ids = targets_df.identifier.values
    target_labels = targets_df.description.values

    # Match tissue targets
    match_tis = []
    for ti, (tid, tlab) in enumerate(zip(target_ids, target_labels)):
        if "GTEX" in tid and keyword in tlab:
            if not (keyword == "blood" and "vessel" in tlab):
                if verbose:
                    print(ti, tid, tlab)
                match_tis.append(ti)

    if len(match_tis) == 0:
        raise ValueError(f"WARNING: No targets matched keyword '{keyword}'.")

    return np.array(match_tis, dtype=int)


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
    match_tis = _match_tissue_targets(gtex_scores_file, keyword, verbose)

    # Load HDF5 data and mappings
    data = _load_hdf5_data(gtex_scores_file, score_key)

    # Reference allele map for flipping
    snp_ref = (
        dict(zip(data["snps"], data["ref_alleles"])) if data["ref_alleles"] else {}
    )

    scores_out = []
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
            else:
                sgs = 0.0
        else:
            sgs = 0.0

        # flip sign if allele1 != reference
        if sgs != 0.0 and si is not None and data["ref_alleles"]:
            ref_a = snp_ref[variant]
            if ref_a != eqtl.allele1:
                sgs *= -1
        scores_out.append(sgs)

    eqtl_df["score"] = scores_out
    return eqtl_df


def _aggregate_scores_across_genes(data: dict, match_tis: np.ndarray):
    """Aggregate absolute scores across all genes for each SNP.

    Args:
        data: Data dictionary from _load_hdf5_data()
        match_tis: Array of target indices that match the tissue

    Returns:
        Dictionary mapping SNP -> aggregated score
    """
    snp_scores = {snp: 0.0 for snp in data["snps"]}

    for (si, gi), row in data["pair_row"].items():
        snp = data["snps"][si]
        vals = data["stat_matrix"][row, match_tis].astype("float32")
        score = float(np.mean(vals))
        snp_scores[snp] += np.abs(score)

    return snp_scores


def classify_auroc(gtex_scores_file: str, keyword: str, score_key: str = "logSUM"):
    """Read eQTL RNA predictions for negatives from the given tissue.

    Args:
      gtex_scores_file (str): Variant scores HDF5.
      tissue_keyword (str): tissue keyword, for matching GTEx targets
      score_key (str): score key in HDF5 file
      verbose (bool): Print matching targets.

    Returns:
      class_auroc (float): Classification AUROC.
    """
    # Match tissue targets
    match_tis = _match_tissue_targets(gtex_scores_file, keyword)

    # Score positives using all genes
    data_pos = _load_hdf5_data(gtex_scores_file, score_key)
    psnp_scores = _aggregate_scores_across_genes(data_pos, match_tis)

    # Score negatives
    gtex_nscores_file = gtex_scores_file.replace("_pos", "_neg")
    data_neg = _load_hdf5_data(gtex_nscores_file, score_key)
    nsnp_scores = _aggregate_scores_across_genes(data_neg, match_tis)

    # compute AUROC
    Xp = list(psnp_scores.values())
    Xn = list(nsnp_scores.values())
    X = Xp + Xn
    y = [1] * len(Xp) + [0] * len(Xn)

    return roc_auc_score(y, X)


def trim_dot(gene_id):
    """Trim dot off GENCODE id's."""
    dot_i = gene_id.rfind(".")
    if dot_i != -1:
        gene_id = gene_id[:dot_i]
    return gene_id


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
