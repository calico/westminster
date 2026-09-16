#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from westminster.bootstrap import (
    MIN_POS,
    N_BOOT,
    cluster_stats,
    collapse_stats,
    cut_by_abs,
    bin_labels,
    per_tissue_metric_by_bin,
    per_tissue_spearman_by_bin,
)
from westminster.gtex import load_qtl_pools

"""
westminster_qtl_cmp

Compare two variant score sets on a fine-mapped QTL benchmark, testing with a
variant-level bootstrap. A QTL recurs across tissues carrying nearly the same
effect size and drawing nearly the same prediction, so an across-tissue test
counts near-duplicates as replicates; the variant is the unit of replication.
"""

# Attribute to stratify by, and its bins. A categorical column passes its level
# order; a numeric one passes edges. 'overall' is the single-bin case, so the
# unstratified comparison takes the same code path as a stratified one.
STRATA = {
    "overall": ("_all", ["all"]),
    "REGION": ("REGION", ["TSS", "CDS", "UTR3"]),
    "TSSD": ("TSSD", [0, 3_000, 20_000, 100_000, 400_000]),
}

# INFO fields carried by the positives and broadcast to their matched negatives.
POSITIVE_ONLY = ["PIP", "REGION", "TSSD", "AFC", "NLP"]


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Compare two variant score sets with a variant-level bootstrap."
    )
    parser.add_argument(
        "-g",
        "--gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_v11/snp/eqtl",
        help="QTL benchmark VCF directory [Default: %(default)s]",
    )
    parser.add_argument("-l", "--labels", help="Comma-separated model labels")
    parser.add_argument("-o", "--out_dir", default="qtl_cmp", help="Output directory")
    parser.add_argument(
        "--strata",
        default="overall,REGION",
        help=f"Comma-separated strata from {sorted(STRATA)} [Default: %(default)s]",
    )
    parser.add_argument(
        "-u",
        "--unit",
        default="tissue",
        choices=["tissue", "variant"],
        help="'tissue' scores per-tissue cells inside each resample; 'variant' "
        "collapses each variant to one row first, for thin strata. "
        "[Default: %(default)s]",
    )
    parser.add_argument(
        "--agg",
        default="median",
        choices=["median", "mean"],
        help="How each draw combines its per-tissue values [Default: %(default)s]",
    )
    parser.add_argument(
        "--cor_col",
        default="coef",
        help="Measured effect size column for Spearman rho [Default: %(default)s]",
    )
    parser.add_argument(
        "-n", "--n_boot", default=N_BOOT, type=int, help="Bootstrap resamples"
    )
    parser.add_argument(
        "--min_n",
        default=MIN_POS,
        type=int,
        help="Positives a (tissue, bin) cell needs to be scored",
    )
    parser.add_argument("--seed", default=0, type=int, help="Bootstrap seed")
    parser.add_argument("metrics_dir1")
    parser.add_argument("metrics_dir2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dirs = [args.metrics_dir1, args.metrics_dir2]
    if args.labels is None:
        label1, label2 = [os.path.basename(d.rstrip("/")) for d in dirs]
    else:
        label1, label2 = args.labels.split(",")

    tissues, pools = load_qtl_pools(args.gtex_vcf_dir, POSITIVE_ONLY, *dirs)
    print(f"Found {len(tissues)} shared tissues")
    for pool in pools:
        for df in pool.values():
            df["_all"] = "all"

    # rho needs a measured effect size; sQTL/paQTL tables may not carry one
    do_cor = all(args.cor_col in df for pool in pools for df in pool.values())
    if not do_cor:
        print(f"Skipping Spearman rho: no '{args.cor_col}' column")

    for name in args.strata.split(","):
        col, edges = STRATA[name]
        tbl = (
            variant_unit_table(pools, col, edges, args, do_cor)
            if args.unit == "variant"
            else tissue_unit_table(pools, col, edges, label1, label2, args, do_cor)
        )
        tbl.to_csv(f"{args.out_dir}/{name}.tsv", sep="\t", float_format="%.4g")
        print(f"\n{name}  ({label1} -> {label2}, {args.unit} unit)")
        print(tabulate(tbl, headers="keys", tablefmt="github", floatfmt=".4g"))


def boot_ci(boots):
    """(2.5, 97.5) percentiles of the finite resamples."""
    finite = np.asarray(boots)[np.isfinite(boots)]
    if finite.size == 0:
        return np.nan, np.nan
    return tuple(np.quantile(finite, [0.025, 0.975]))


def tissue_unit_table(pools, col, edges, label1, label2, args, do_cor):
    """Per-tissue cells aggregated across tissues, bootstrapped over variants."""
    order = bin_labels(edges)
    common = dict(
        col=col,
        edges=edges,
        min_n=args.min_n,
        agg=args.agg,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    rows = []

    clf = cluster_stats(pools, kind="clf", **common)
    for met in ("AUROC", "AUPRC"):
        levels = [
            [
                per_tissue_metric_by_bin(df, col, edges, met.lower(), min_n=args.min_n)
                for df in pool.values()
            ]
            for pool in pools
        ]
        for b in order:
            lo, hi = boot_ci(clf.attrs["boots"][(b, met)])
            rows.append(
                {
                    "bin": b,
                    "metric": met,
                    label1: np.nanmedian([s.get(b, np.nan) for s in levels[0]]),
                    label2: np.nanmedian([s.get(b, np.nan) for s in levels[1]]),
                    "delta": clf.loc[(b, met), "delta"],
                    "lo": lo,
                    "hi": hi,
                    "p": clf.loc[(b, met), "p"],
                }
            )

    if do_cor:
        cor = cluster_stats(pools, kind="cor", cor_col=args.cor_col, **common)
        levels = [
            [
                per_tissue_spearman_by_bin(
                    df, col, edges, y_col=args.cor_col, min_n=args.min_n
                )
                for df in pool.values()
            ]
            for pool in pools
        ]
        for b in order:
            lo, hi = boot_ci(cor.attrs["boots"][b])
            rows.append(
                {
                    "bin": b,
                    "metric": "rho",
                    label1: np.nanmedian([s.get(b, np.nan) for s in levels[0]]),
                    label2: np.nanmedian([s.get(b, np.nan) for s in levels[1]]),
                    "delta": cor.loc[b, "delta"],
                    "lo": lo,
                    "hi": hi,
                    "p": cor.loc[b, "p"],
                }
            )

    return with_counts(pd.DataFrame(rows), pools[1], col, edges)


def variant_unit_table(pools, col, edges, args, do_cor):
    """Each variant collapsed to one row, then bootstrapped."""
    rows = []
    for met in ["auroc", "auprc"] + (["spearman"] if do_cor else []):
        for b in bin_labels(edges):
            res = collapse_stats(
                pools,
                bin_selector(col, edges, b),
                metric=met,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            rows.append(
                {
                    "bin": b,
                    "metric": met.upper() if met != "spearman" else "rho",
                    **{k: res[k] for k in ("m1", "m2", "delta", "lo", "hi", "p")},
                }
            )
    return with_counts(pd.DataFrame(rows), pools[1], col, edges)


def bin_selector(col, edges, label):
    """DataFrame -> the rows of one bin, for `collapse_stats`."""

    def select(df):
        return df[cut_by_abs(df[col], edges)[0].to_numpy() == label]

    return select


def with_counts(tbl, pool, col, edges):
    """Append positive-row and unique-positive-variant counts per bin."""
    allrows = pd.concat(pool.values()).rename_axis("variant").reset_index()
    allrows["bin"], _ = cut_by_abs(allrows[col], edges)
    pos = allrows[allrows.label == "pos"]
    tbl["n_rows"] = tbl["bin"].map(pos.groupby("bin", observed=True).size())
    tbl["n_var"] = tbl["bin"].map(
        pos.groupby("bin", observed=True)["variant"].nunique()
    )
    return tbl.set_index(["bin", "metric"])


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
