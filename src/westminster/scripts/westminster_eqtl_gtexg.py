#!/usr/bin/env python
import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from westminster.gtex import (
    match_tissue_targets,
    read_gene_tss,
    tissue_keywords,
    trim_dot,
    variant_pos,
)

"""
westminster_eqtl_gtexg.py

Score variant-effect predictions against GTEx eQTLs across multiple metrics.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--genes_bed_file",
        default=f"{os.environ['HG38']}/genes/gencode48/gencode48_basic_tss2.bed",
        help="BED file of gene TSS positions, for computing variant distance to TSS",
    )
    parser.add_argument(
        "-g",
        "--gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_v11/eqtl_pip90",
        help="GTEx VCF directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="metrics_out",
        help="Output directory for tissue metrics",
    )
    parser.add_argument(
        "-s",
        "--snp_stat",
        default="covgene/logFC",
        help="SNP statistic. [Default: %(default)s]",
    )
    parser.add_argument(
        "--ems",
        action="store_true",
        help="Use the legacy EMS pipeline: pull variant metadata from the "
        "gtex_fine/tissues_susie/{tissue}.tsv tables instead of the VCF INFO.",
    )
    parser.add_argument(
        "--egene_window",
        type=int,
        default=384_000,
        help="Cis window (+/- bp around variant) for enumerating eGene "
        "candidates. Defaults to half the new-model sequence length (~768 kb); "
        "candidates beyond the model's predictable window get no score and are "
        "excluded from metrics.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("gtex_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gene_tss = read_gene_tss(args.genes_bed_file)
    chrom_tss = _index_tss_by_chrom(gene_tss)

    metrics_rows = []
    egene_metrics_rows = []
    egene_dist_rows = []

    for tissue, keyword in tissue_keywords.items():
        if args.verbose:
            print(tissue)

        # read causal variants
        tissue_vcf_file = f"{args.gtex_vcf_dir}/{tissue}_pos.vcf"
        if args.ems:
            eqtl_df = read_eqtl_ems(tissue, args.gtex_vcf_dir)
        else:
            eqtl_df = read_eqtl_vcf(tissue_vcf_file)
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

            # score negatives at their assigned gene (INFO GENE=) on the neg
            # HDF5, exactly as positives are scored at their eQTL gene
            gtex_nscores_file = gtex_scores_file.replace("_pos", "_neg")
            neg_df = read_neg_vcf(f"{args.gtex_vcf_dir}/{tissue}_neg.vcf")
            if neg_df is None:
                print(f"Skipping {tissue}: no negatives.", file=sys.stderr)
                continue
            neg_df = add_scores(
                gtex_nscores_file, keyword, neg_df, args.snp_stat, verbose=args.verbose
            )
            neg_meas_df = neg_df[neg_df.measured]

            # classification AUROC/AUPRC: |gene-specific logFC| at each
            # variant's single designated gene, positives vs negatives
            Xp = np.abs(eqtl_df.score.values)
            Xn = np.abs(neg_df.score.values)
            class_labels = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
            class_scores = np.r_[Xp, Xn]
            class_auroc = roc_auc_score(class_labels, class_scores)
            class_auprc = average_precision_score(class_labels, class_scores)

            # measured classification: restrict each side to in-window variants
            Xp_m = np.abs(eqtl_meas_df.score.values)
            Xn_m = np.abs(neg_meas_df.score.values)
            class_labels_meas = np.r_[np.ones(len(Xp_m)), np.zeros(len(Xn_m))]
            class_scores_meas = np.r_[Xp_m, Xn_m]
            class_auroc_meas = roc_auc_score(class_labels_meas, class_scores_meas)
            class_auprc_meas = average_precision_score(
                class_labels_meas, class_scores_meas
            )

            # write combined variant table: one designated-gene score per
            # variant (pos = eQTL gene, neg = assigned gene)
            pos_var_df = eqtl_df.rename(columns={"score": "pred"}).assign(label="pos")
            neg_var_df = neg_df.rename(columns={"score": "pred"}).assign(
                label="neg", coef=np.nan
            )
            scatter_cols = ["variant", "label", "coef", "pred", "measured", "tss_dist"]
            scatter_df = pd.concat(
                [pos_var_df[scatter_cols], neg_var_df[scatter_cols]],
                ignore_index=True,
            )
            scatter_df.to_csv(f"{args.out_dir}/{tissue}.tsv", index=False, sep="\t")

            # eGene assignment evaluation
            tissue_egene, dist_rows = egene_evaluate(
                tissue=tissue,
                eqtl_df=eqtl_df,
                gtex_scores_file=gtex_scores_file,
                keyword=keyword,
                score_key=args.snp_stat,
                chrom_tss=chrom_tss,
                window=args.egene_window,
                out_dir=f"{args.out_dir}/egene",
            )
            if tissue_egene is not None:
                egene_metrics_rows.append(tissue_egene)
                egene_dist_rows.extend(dist_rows)

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

    egene_dir = f"{args.out_dir}/egene"
    os.makedirs(egene_dir, exist_ok=True)
    egene_df = pd.DataFrame(egene_metrics_rows)
    egene_df.to_csv(
        f"{egene_dir}/metrics.tsv", sep="\t", index=False, float_format="%.4f"
    )
    if egene_dist_rows:
        pd.DataFrame(egene_dist_rows).to_csv(
            f"{egene_dir}/metrics_by_distance.tsv",
            sep="\t",
            index=False,
            float_format="%.4f",
        )
    if not egene_df.empty:
        print("eGene AUROC:        %.4f" % np.nanmean(egene_df.auroc))
        print("eGene AUPRC:        %.4f" % np.nanmean(egene_df.auprc))
        print("eGene top-1:        %.4f" % np.nanmean(egene_df.top1))
        print("eGene top-3:        %.4f" % np.nanmean(egene_df.top3))
        print("Distance AUROC:     %.4f" % np.nanmean(egene_df.baseline_auroc))
        print("Distance top-1:     %.4f" % np.nanmean(egene_df.baseline_top1))


def read_eqtl_vcf(tissue_vcf_file: str):
    """Read eQTLs from the new gtex_eqtl VCF (INFO carries GENE, AFC, TSSD)."""
    if not os.path.isfile(tissue_vcf_file):
        return None
    rows = []
    for line in open(tissue_vcf_file):
        if line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        vid = cols[2]
        ref = cols[3]
        info = dict(f.split("=", 1) for f in cols[7].split(";") if "=" in f)
        gene = trim_dot(info.get("GENE", ""))
        afc = float(info["AFC"])
        tssd_raw = info.get("TSSD", ".")
        tssd = np.nan if tssd_raw == "." else float(tssd_raw)
        if not gene:
            continue
        rows.append(
            {
                "variant": vid,
                "gene": gene,
                "coef": afc,
                "allele1": ref,
                "tss_dist": tssd,
            }
        )
    return pd.DataFrame(rows) if rows else None


def read_neg_vcf(neg_vcf_file: str):
    """Read matched-negative variants from a gtex_eqtl {tissue}_neg.vcf.

    Negatives carry an assigned gene (INFO GENE=, the nearest-cognate-feature
    reassignment) but no AFC. Returns one row per variant with the columns
    add_scores() needs, so each negative is scored at its assigned gene exactly
    as a positive is scored at its eQTL gene.
    """
    if not os.path.isfile(neg_vcf_file):
        return None
    rows = []
    for line in open(neg_vcf_file):
        if line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        vid = cols[2]
        ref = cols[3]
        info = dict(f.split("=", 1) for f in cols[7].split(";") if "=" in f)
        gene = trim_dot(info.get("GENE", ""))
        tssd_raw = info.get("TSSD", ".")
        tssd = np.nan if tssd_raw == "." else float(tssd_raw)
        if not gene:
            continue
        rows.append({"variant": vid, "gene": gene, "allele1": ref, "tss_dist": tssd})
    return pd.DataFrame(rows) if rows else None


def read_eqtl_ems(tissue: str, gtex_vcf_dir: str, pip_t: float = 0.9):
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
                "tss_dist": np.nan,
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


def _index_tss_by_chrom(gene_tss: dict):
    """Bucket gene_tss by chromosome and sort by TSS position for cis lookups.

    Args:
        gene_tss: {trimmed_ensg: (chrom, tss_pos)} from read_gene_tss().

    Returns:
        {chrom: (positions_array, gene_ids_array)} both sorted by position.
    """
    by_chrom = {}
    for ensg, (chrom, tss) in gene_tss.items():
        by_chrom.setdefault(chrom, []).append((tss, ensg))
    out = {}
    for chrom, entries in by_chrom.items():
        entries.sort(key=lambda x: x[0])
        positions = np.array([e[0] for e in entries], dtype=np.int64)
        genes = np.array([e[1] for e in entries], dtype=object)
        out[chrom] = (positions, genes)
    return out


# Distance bins (lower-inclusive, upper-exclusive) for stratified eGene metrics.
EGENE_DIST_BINS = [
    ("0_1k", 0, 1_000),
    ("1k_10k", 1_000, 10_000),
    ("10k_50k", 10_000, 50_000),
    ("50k_100k", 50_000, 100_000),
    ("100k_200k", 100_000, 200_000),
    ("200k_400k", 200_000, 400_000),
]


def egene_evaluate(
    tissue: str,
    eqtl_df: pd.DataFrame,
    gtex_scores_file: str,
    keyword: str,
    score_key: str,
    chrom_tss: dict,
    window: int,
    out_dir: str,
):
    """Evaluate eGene assignment for one tissue.

    For each fine-mapped variant, enumerate all gene candidates in `chrom_tss`
    whose TSS falls within +/- `window` bp, label each (variant, gene) by
    whether it's the true eGene, score with the model, and report AUROC/AUPRC
    (pooled and per distance bin) plus per-variant top-1/top-3 accuracy.

    Args:
        tissue: tissue name (used in output filenames + metrics row).
        eqtl_df: rows of (variant, gene, ...) from read_eqtl_vcf — the
            positives.
        gtex_scores_file: path to gene-pair-indexed HDF5 for this tissue.
        keyword: tissue keyword for matching model targets.
        score_key: HDF5 stat to read (e.g. "gene/logFC").
        chrom_tss: output of _index_tss_by_chrom — sorted TSS table per chrom.
        window: cis half-window in bp.
        out_dir: directory for {tissue}_pairs.tsv (created if missing).

    Returns:
        (metrics_row, dist_rows): per-tissue summary dict and a list of per-bin
        rows. (None, []) if the tissue has no usable scored pairs.
    """
    os.makedirs(out_dir, exist_ok=True)

    match_tis = _match_tissue_targets(gtex_scores_file, keyword, score_key)
    data = _load_hdf5_data(gtex_scores_file, score_key)

    # positives: {(variant, gene_trim)} — variant may have multiple eGenes
    positive_pairs = set(zip(eqtl_df.variant, eqtl_df.gene))
    variants = eqtl_df.variant.unique()

    pair_rows = []
    for variant in variants:
        chrom, vpos = variant_pos(variant)
        if chrom not in chrom_tss:
            continue
        positions, gene_arr = chrom_tss[chrom]
        lo = np.searchsorted(positions, vpos - window, side="left")
        hi = np.searchsorted(positions, vpos + window, side="right")
        if lo == hi:
            continue

        si = data["snp_to_index"].get(variant)
        for j in range(lo, hi):
            gene = str(gene_arr[j])
            tss = int(positions[j])
            tss_dist = vpos - tss  # signed (positive => downstream of TSS)
            label = 1 if (variant, gene) in positive_pairs else 0
            score = np.nan
            in_window = False
            if si is not None:
                gi = data["trimmed_to_index"].get(gene)
                if gi is not None:
                    row = data["pair_row"].get((si, gi))
                    if row is not None:
                        vals = data["stat_matrix"][row, match_tis].astype("float32")
                        score = float(np.abs(np.mean(vals)))
                        in_window = True
            pair_rows.append(
                {
                    "variant": variant,
                    "gene": gene,
                    "tss_dist": tss_dist,
                    "abs_tss_dist": abs(tss_dist),
                    "label": label,
                    "score": score,
                    "in_window": in_window,
                    "baseline": 1.0 / (abs(tss_dist) + 1),
                }
            )

    if not pair_rows:
        return None, []

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(
        f"{out_dir}/{tissue}_pairs.tsv", sep="\t", index=False, float_format="%.6g"
    )

    # pooled metrics on scored pairs only
    scored = pairs_df[pairs_df.in_window]
    if scored.empty:
        return None, []
    auroc, auprc = _safe_binary_metrics(scored.label, scored.score)
    base_auroc, base_auprc = _safe_binary_metrics(scored.label, scored.baseline)
    top1 = _topk_accuracy(scored, "score", 1)
    top3 = _topk_accuracy(scored, "score", 3)
    base_top1 = _topk_accuracy(scored, "baseline", 1)
    base_top3 = _topk_accuracy(scored, "baseline", 3)

    metrics_row = {
        "tissue": tissue,
        "n_variants": int(scored.variant.nunique()),
        "n_pairs": int(len(pairs_df)),
        "n_scored": int(len(scored)),
        "n_pos": int(scored.label.sum()),
        "coverage": float(len(scored) / len(pairs_df)),
        "auroc": auroc,
        "auprc": auprc,
        "top1": top1,
        "top3": top3,
        "baseline_auroc": base_auroc,
        "baseline_auprc": base_auprc,
        "baseline_top1": base_top1,
        "baseline_top3": base_top3,
    }

    dist_rows = []
    for label, lo, hi in EGENE_DIST_BINS:
        sub = scored[(scored.abs_tss_dist >= lo) & (scored.abs_tss_dist < hi)]
        b_auroc, b_auprc = _safe_binary_metrics(sub.label, sub.score)
        d_auroc, d_auprc = _safe_binary_metrics(sub.label, sub.baseline)
        dist_rows.append(
            {
                "tissue": tissue,
                "bin": label,
                "n_pairs": int(len(sub)),
                "n_pos": int(sub.label.sum()),
                "auroc": b_auroc,
                "auprc": b_auprc,
                "baseline_auroc": d_auroc,
                "baseline_auprc": d_auprc,
            }
        )

    return metrics_row, dist_rows


def _safe_binary_metrics(labels, scores):
    """Return (AUROC, AUPRC) or (NaN, NaN) if labels lack both classes."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    mask = ~np.isnan(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(labels) == 0 or labels.min() == labels.max():
        return float("nan"), float("nan")
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def _topk_accuracy(scored: pd.DataFrame, score_col: str, k: int):
    """Fraction of variants whose top-k candidates (by score_col) include a positive.

    Variants with no positive among scored candidates are skipped.
    """
    hits = 0
    n = 0
    for _, group in scored.groupby("variant", sort=False):
        if group.label.sum() == 0:
            continue
        top = group.nlargest(k, score_col)
        n += 1
        if top.label.sum() > 0:
            hits += 1
    return float(hits / n) if n else float("nan")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
