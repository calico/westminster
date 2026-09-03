#!/usr/bin/env python
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from westminster.gtex import gtex_keywords, match_tissue_targets, read_targets, trim_dot

"""
westminster_eqtl_spec

Cross-tissue specificity of predicted eQTL effects: for each fine-mapped
(variant, gene) pair, does the model put the effect in the right tissues?

The model resolves coarse tissue groups (brain, heart, ...), not GTEx's 49 fine
tissues, so both predictions and measurements are collapsed onto that axis. Per
pair we form a model vector over groups (mean logFC across the group's GTEx
tracks, oriented ref->alt) and compare it to two measurements: which groups the
pair is fine-mapped in, and its tensorQTL slope in each group.
"""

# n_sig strata (lower-inclusive, upper-exclusive) for the summary table.
NSIG_BINS = [("1", 1, 2), ("2", 2, 3), ("3", 3, 4), ("4_5", 4, 6),
             ("6_9", 6, 10), ("10+", 10, 10**9)]

METRICS = ["spec_auroc", "tpm_auroc", "raw_auroc", "slope_spearman", "sig_spearman"]

TPM_EXPRESSED = 1.0  # median TPM above which a gene counts as expressed in a group


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Score cross-tissue specificity of predicted eQTL effects."
    )
    parser.add_argument(
        "-g",
        "--gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_v11/current/snp/eqtl",
        help="GTEx VCF directory, holding *_pos.vcf and spec_slopes.parquet",
    )
    parser.add_argument(
        "-o", "--out_dir", default="spec_out", help="Output directory"
    )
    parser.add_argument(
        "-s",
        "--snp_stat",
        default="covgene/logFC",
        help="Pair-indexed SNP statistic. [Default: %(default)s]",
    )
    parser.add_argument(
        "--tpm_gct",
        default=None,
        help="GTEx median TPM GCT, enabling the gene-expression baseline",
    )
    parser.add_argument(
        "--min_sig",
        type=int,
        default=2,
        help="Minimum significant groups for sig_spearman. [Default: %(default)s]",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Null control: permute each pair's model vector across groups",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("gtex_dir", help="Iteration dir holding merge_pos/scores.h5")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    merge_file = f"{args.gtex_dir}/merge_pos/scores.h5"

    groups, group_cols = resolve_groups(merge_file, args.snp_stat, args.verbose)
    pd.DataFrame(
        {"group": groups, "tracks": [len(group_cols[g]) for g in groups]}
    ).to_csv(f"{args.out_dir}/groups.tsv", sep="\t", index=False)
    print(f"Tissue groups: {len(groups)}")

    pos_df, sig = read_positives(args.gtex_vcf_dir, groups)
    print(f"Positive pairs: {len(pos_df):,}")

    pairs_df, model = model_matrix(
        merge_file, args.snp_stat, group_cols, pos_df, args.shuffle
    )
    sig = sig[pairs_df.index.values]
    pairs_df = pairs_df.reset_index(drop=True)
    print(f"Scored pairs:   {len(pairs_df):,} ({len(pairs_df) / len(pos_df):.1%})")

    slope = slope_matrix(
        f"{args.gtex_vcf_dir}/spec_slopes.parquet", pairs_df, groups
    )
    tpm = tpm_matrix(args.tpm_gct, pairs_df, groups) if args.tpm_gct else None
    if tpm is not None:
        pairs_df["expressed_all"] = (tpm >= TPM_EXPRESSED).all(axis=1)

    # Divide out each group's amplitude so that within-pair rankings compare
    # tissues rather than track scale, which varies ~2x across groups.
    for name, values in [
        ("spec_auroc", group_scale(model)),
        ("raw_auroc", model),
        ("tpm_auroc", group_scale(tpm) if tpm is not None else None),
    ]:
        pairs_df[name] = (
            pair_auroc(np.abs(values), sig) if values is not None else np.nan
        )
    # zero-fill only after scaling: a group's amplitude must come from the cells
    # where its slope was measured, not from the absent ones
    model_n = group_scale(model)
    slope_n = np.nan_to_num(group_scale(slope))
    pairs_df["slope_spearman"] = pair_spearman(model_n, slope_n)
    pairs_df["sig_spearman"] = pair_spearman(model_n, slope_n, mask=sig)
    pairs_df.loc[pairs_df.n_sig < args.min_sig, "sig_spearman"] = np.nan

    pairs_df.to_csv(
        f"{args.out_dir}/pairs.tsv", sep="\t", index=False, float_format="%.6g"
    )
    np.savez_compressed(
        f"{args.out_dir}/vectors.npz",
        groups=np.array(groups),
        variant=pairs_df.variant.values,
        gene=pairs_df.gene.values,
        model=model,
        slope=slope,
        sig=sig,
        **({"tpm": tpm} if tpm is not None else {}),
    )

    metrics_df = summarize(pairs_df, len(groups))
    metrics_df.to_csv(
        f"{args.out_dir}/metrics.tsv", sep="\t", index=False, float_format="%.4f"
    )
    print(metrics_df.to_string(index=False))


################################################################################
# inputs
################################################################################
def resolve_groups(merge_file: str, score_key: str, verbose: bool = False):
    """Tissue groups the model resolves, and each one's track columns.

    Args:
        merge_file: Path to merge_pos/scores.h5.
        score_key: Pair-indexed statistic, e.g. covgene/logFC.
        verbose: Print each group's matched targets.

    Returns:
        (list[str], dict[str, np.ndarray]): Sorted group names, and group ->
            target column indices. Keywords matching no track are dropped.
    """
    targets_df, gene_targets = read_targets(merge_file, score_key)
    group_cols = {}
    for keyword in sorted(set(gtex_keywords.values())):
        match_tis = match_tissue_targets(targets_df, keyword, gene_targets, verbose)
        if len(match_tis) > 0:
            group_cols[keyword] = match_tis
        else:
            print(f"Dropping {keyword}: no matching targets.", file=sys.stderr)
    if len(group_cols) < 2:
        raise ValueError("Fewer than 2 resolvable tissue groups.")
    return sorted(group_cols), group_cols


def _float(value):
    """Parse a VCF INFO number, mapping absent and '.' to NaN."""
    return float(value) if value not in (None, ".") else np.nan


def _nanmean(values):
    """Mean of the non-NaN entries, or NaN if a pair has none."""
    present = [v for v in values if not np.isnan(v)]
    return float(np.mean(present)) if present else np.nan


def read_positives(gtex_vcf_dir: str, groups: list):
    """Read fine-mapped positives from the per-tissue VCFs, pooled by group.

    Args:
        gtex_vcf_dir: Directory of {tissue}_pos.vcf files.
        groups: Tissue group axis.

    Returns:
        (pd.DataFrame, np.ndarray): One row per unique (variant, gene) pair with
            n_sig, maf, tss_dist; and a (pairs, groups) boolean significance mask.
    """
    group_idx = {g: i for i, g in enumerate(groups)}
    pair_sig, pair_maf, pair_tssd = {}, {}, {}
    for pos_vcf in sorted(glob.glob(f"{gtex_vcf_dir}/*_pos.vcf")):
        for line in open(pos_vcf):
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            info = dict(f.split("=", 1) for f in cols[7].split(";") if "=" in f)
            gene = info.get("GENE", "")
            gi = group_idx.get(gtex_keywords.get(info.get("TISSUE", "")), None)
            if not gene or gi is None:
                continue
            key = (cols[2], gene)
            pair_sig.setdefault(key, set()).add(gi)
            pair_maf.setdefault(key, []).append(_float(info.get("MAF")))
            pair_tssd.setdefault(key, []).append(_float(info.get("TSSD")))

    keys = sorted(pair_sig)
    sig = np.zeros((len(keys), len(groups)), dtype=bool)
    for pi, key in enumerate(keys):
        sig[pi, list(pair_sig[key])] = True
    pos_df = pd.DataFrame(
        {
            "variant": [k[0] for k in keys],
            "gene": [k[1] for k in keys],
            "n_sig": sig.sum(axis=1),
            "maf": [_nanmean(pair_maf[k]) for k in keys],
            "tss_dist": [_nanmean(pair_tssd[k]) for k in keys],
        }
    )
    return pos_df, sig


def model_matrix(
    merge_file: str,
    score_key: str,
    group_cols: dict,
    pos_df: pd.DataFrame,
    shuffle: bool = False,
):
    """Per-pair model scores averaged within each tissue group, oriented ref->alt.

    Only the rows for requested pairs are read, and each is reduced to one value
    per group on the fly; the full (pairs, tracks) matrix is never materialized.

    Args:
        merge_file: Path to merge_pos/scores.h5.
        score_key: Pair-indexed statistic.
        group_cols: group -> target column indices, from resolve_groups.
        pos_df: Positive pairs, from read_positives.
        shuffle: Permute each pair's group vector, as a null control.

    Returns:
        (pd.DataFrame, np.ndarray): pos_df restricted to pairs the model scored,
            retaining its index; and the matching (pairs, groups) score matrix.
    """
    groups = sorted(group_cols)
    with h5py.File(merge_file, "r") as h5_file:
        snp_to_index = {
            s.decode(): i for i, s in enumerate(h5_file["snp"][:])
        }
        gene_to_index = {}
        for gi, gene in enumerate(h5_file["gene_ids"][:]):
            gene_to_index.setdefault(trim_dot(gene.decode()), gi)
        pair_row = {
            (int(si), int(gi)): row
            for row, (si, gi) in enumerate(
                zip(h5_file["snp_idx"][:], h5_file["gene_idx"][:])
            )
        }
        ref_allele = [a.decode() for a in h5_file["ref_allele"][:]]

        rows, keep = [], []
        for pi, variant, gene in pos_df[["variant", "gene"]].itertuples():
            si = snp_to_index.get(variant)
            gi = gene_to_index.get(gene)
            row = None if si is None or gi is None else pair_row.get((si, gi))
            if row is not None:  # else the gene is outside the model's window
                rows.append(row)
                keep.append(pi)

        # h5py fancy selection requires increasing indices; restore pair order after
        order = np.argsort(rows, kind="stable")
        scores = h5_file[score_key][np.array(rows)[order].tolist()].astype("float32")
        scores = scores[np.argsort(order, kind="stable")]

    model = np.stack([scores[:, group_cols[g]].mean(axis=1) for g in groups], axis=1)

    pairs_df = pos_df.loc[keep]
    flip = np.array(
        [
            ref_allele[snp_to_index[v]] != v.split("_")[2]
            for v in pairs_df.variant.values
        ]
    )
    model[flip] *= -1

    if shuffle:
        rng = np.random.default_rng(0)
        model = np.stack([rng.permutation(row) for row in model])
    return pairs_df, model


def slope_matrix(slopes_file: str, pairs_df: pd.DataFrame, groups: list):
    """Mean tensorQTL slope per (pair, group), NaN where there is no signal.

    NaN rather than zero because the two mean different things to group_scale:
    "no effect measured here" must not contribute to a group's amplitude. Callers
    fill with zero after scaling.

    Args:
        slopes_file: spec_slopes.parquet from make_vcfs.build_spec_slopes.
        pairs_df: Scored pairs, in output order.
        groups: Tissue group axis.

    Returns:
        np.ndarray: (pairs, groups) slope matrix, NaN where not significant.
    """
    slope_df = pd.read_parquet(
        slopes_file, columns=["variant_id", "gene", "tissue", "slope"]
    )
    slope_df["group"] = slope_df.tissue.map(gtex_keywords)
    slope_df = slope_df[slope_df.group.isin(groups)]
    mean_slope = slope_df.groupby(["variant_id", "gene", "group"]).slope.mean()

    pair_index = {(v, g): i for i, (v, g) in enumerate(
        zip(pairs_df.variant.values, pairs_df.gene.values)
    )}
    group_index = {g: i for i, g in enumerate(groups)}
    slope = np.full((len(pairs_df), len(groups)), np.nan, dtype="float32")
    for (variant, gene, group), value in mean_slope.items():
        pi = pair_index.get((variant, gene))
        if pi is not None:
            slope[pi, group_index[group]] = value
    return slope


def tpm_matrix(tpm_gct: str, pairs_df: pd.DataFrame, groups: list):
    """Mean GTEx median TPM per (pair's gene, group).

    Args:
        tpm_gct: GTEx median TPM GCT (2 header lines, then Name/Description/tissues).
        pairs_df: Scored pairs, in output order.
        groups: Tissue group axis.

    Returns:
        np.ndarray: (pairs, groups) expression matrix, NaN for genes absent from
            the GCT.
    """
    tpm_df = pd.read_csv(tpm_gct, sep="\t", skiprows=2, index_col=0).drop(
        columns="Description"
    )
    tpm_df.index = [trim_dot(g) for g in tpm_df.index]
    tpm_df = tpm_df[~tpm_df.index.duplicated()]
    tpm_df.columns = [gtex_keywords.get(c) for c in tpm_df.columns]
    group_tpm = tpm_df.loc[:, tpm_df.columns.notna()].T.groupby(level=0).mean().T
    return group_tpm.reindex(index=pairs_df.gene.values, columns=groups).values


################################################################################
# metrics
################################################################################
def group_scale(values: np.ndarray):
    """Divide each group column by its amplitude across pairs.

    Group score distributions differ ~2x in scale, so an un-normalized within-pair
    ranking partly reflects which tissues the model was trained on most deeply.
    Dividing by a robust scale about zero equalizes them while preserving sign.
    NaN marks an unmeasured cell and is excluded from its group's scale.

    Args:
        values: (pairs, groups) matrix, NaN where unmeasured.

    Returns:
        np.ndarray: Rescaled matrix.
    """
    scale = np.nanmedian(np.abs(values), axis=0)
    return values / np.where(scale > 0, scale, 1.0)


def pair_auroc(scores: np.ndarray, sig: np.ndarray):
    """Per-pair AUROC ranking tissue groups by score against the significance mask.

    Computed as the Mann-Whitney rank sum, which vectorizes over pairs where a
    per-pair roc_auc_score call would not.

    Args:
        scores: (pairs, groups) non-negative scores.
        sig: (pairs, groups) boolean significance mask.

    Returns:
        np.ndarray: Per-pair AUROC, NaN where all or no groups are significant.
    """
    n_pos = sig.sum(axis=1)
    n_neg = scores.shape[1] - n_pos
    rank_sum = (rankdata(scores, axis=1) * sig).sum(axis=1)
    valid = (n_pos > 0) & (n_neg > 0) & np.isfinite(scores).all(axis=1)
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / np.where(valid, n_pos * n_neg, 1)
    return np.where(valid, auroc, np.nan)


def pair_spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray = None):
    """Per-pair Spearman correlation across tissue groups.

    Ranking within a pair makes this invariant to the pair's overall effect size,
    so it measures the cross-tissue pattern alone.

    Args:
        x: (pairs, groups) matrix.
        y: (pairs, groups) matrix.
        mask: Optional (pairs, groups) boolean, restricting each pair's groups.

    Returns:
        np.ndarray: Per-pair correlation, NaN where either side is constant.
    """
    out = np.full(len(x), np.nan)
    for pi, (xi, yi) in enumerate(zip(x, y)):
        if mask is not None:
            xi, yi = xi[mask[pi]], yi[mask[pi]]
        if len(xi) > 1 and np.ptp(xi) > 0 and np.ptp(yi) > 0:
            out[pi] = spearmanr(xi, yi)[0]
    return out


def summarize(pairs_df: pd.DataFrame, n_groups: int):
    """Mean of each metric overall, per n_sig stratum, and on the expressed subset.

    Args:
        pairs_df: Per-pair metrics.
        n_groups: Size of the tissue group axis.

    Returns:
        pd.DataFrame: One row per stratum.
    """
    strata = [("all", pairs_df)]
    strata += [
        (label, pairs_df[(pairs_df.n_sig >= lo) & (pairs_df.n_sig < hi)])
        for label, lo, hi in NSIG_BINS
    ]
    if "expressed_all" in pairs_df:
        strata.append(("expressed_all", pairs_df[pairs_df.expressed_all]))

    rows = []
    for label, sub in strata:
        row = {"stratum": label, "n_groups": n_groups, "pairs": len(sub)}
        row.update({m: sub[m].mean() for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
