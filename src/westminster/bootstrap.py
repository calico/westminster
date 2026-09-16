# Copyright 2025 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""Variant-level bootstrap for comparing two models on a QTL benchmark.

A fine-mapped QTL recurs across tissues carrying nearly the same effect size and
drawing nearly the same prediction (measured across-tissue ICC of |pred| 0.87
non-exonic, 0.99 in 3' UTRs), so the per-tissue values an across-tissue
signed-rank test pairs are near-duplicates and its p reports mostly how many
tissues the benchmark has. The variant is the unit of replication.

`variant_bootstrap` is the primitive: resample variants with replacement within
strata, carry every tissue row of a drawn variant with it, hand both models the
same resample, and recompute the statistic on each draw. Two statistics ship on
top of it:

- `cluster_stats` scores each (tissue, bin) cell inside a resample and aggregates
  across tissues. Use it when the strata are thick enough for per-tissue cells.
- `collapse_stats` averages each variant over its tissues first and scores one row
  per variant. Use it when they are not.

Both read the per-tissue tables `westminster_eqtl_gtex` writes, loaded by
`westminster.gtex.load_qtl_pools`.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# One inclusion rule for every per-tissue estimate: a (tissue, bin) cell needs
# at least this many *positives* to be scored. Counting positives rather than
# total rows is what makes the classification and correlation thresholds
# comparable — rho is computed on positives alone, while a classification bin
# also carries its matched negatives, so an identical row threshold is twice as
# permissive on the classification side.
#
# 20 is chosen from the coverage/precision tradeoff over the 1,003 GTEx eQTL
# (tissue x bin) cells: it keeps 94.8% of them (vs 91.2% at 30, 96.5% at 15)
# while holding the null SD of rho to 1/sqrt(19) = 0.23. The cells it drops are
# concentrated in the genuinely thin bins, which is where a 10-variant rho is
# noise rather than signal. Lower it for strata far thinner than a whole
# benchmark's, and prefer `collapse_stats` once you have to.
MIN_POS = 20

# Resamples for the variant bootstrap. The smallest p it can report is 2/n_boot,
# so 2000 is what keeps p < 1e-3 attainable.
N_BOOT = 2000


################################################################################
# metric kernels
################################################################################
def clf_metrics(y, s):
    """(AUROC, AUPRC) for one cell, off a single sort; (nan, nan) if one class.

    Equals sklearn's roc_auc_score and average_precision_score (step, no
    interpolation) to machine precision, skipping their per-call validation —
    the bottleneck in the bootstrap hot loops, which want both numbers anyway.
    Tied scores collapse to one threshold each, which is what keeps a model
    emitting exactly 0 for a fifth of its variants (anything outside its
    receptive field) from being ranked above the ties it is level with.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    order = np.argsort(s, kind="mergesort")[::-1]
    y, s = y[order], s[order]
    at = np.r_[np.where(np.diff(s))[0], len(s) - 1]  # one per distinct score
    tp, fp = np.cumsum(y)[at], np.cumsum(1 - y)[at]
    d_tp, d_fp = np.diff(np.r_[0, tp]), np.diff(np.r_[0, fp])
    # each tied block is one trapezoid on the ROC, one step on the PR curve
    auroc = np.sum(d_fp * (np.r_[0, tp[:-1]] + d_tp / 2)) / (n_pos * n_neg)
    auprc = np.sum(d_tp / n_pos * (tp / (tp + fp)))
    return float(auroc), float(auprc)


def auroc(y, s):
    return clf_metrics(y, s)[0]


def auprc(y, s):
    return clf_metrics(y, s)[1]


_METRIC_FNS = {"auroc": auroc, "auprc": auprc}


def rankdata(a):
    """Average ranks, as scipy.stats.rankdata, without its per-call overhead."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])  # tie-block starts
    mark = np.zeros(a.size, np.int64)
    mark[start] = 1
    grp = np.cumsum(mark) - 1
    out = np.empty(a.size)
    out[order] = (start + 1 + (np.bincount(grp) - 1) / 2)[grp]
    return out


def spearman(x, y):
    """Spearman rho, nan where either variable is constant.

    scipy.stats.spearmanr without its per-call input validation, for the
    bootstrap hot loops.
    """
    if x.size < 2:
        return np.nan
    rx = rankdata(x) - (x.size + 1) / 2
    ry = rankdata(y) - (y.size + 1) / 2
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d else np.nan


################################################################################
# bin bookkeeping
################################################################################
def _is_categorical_bins(bin_edges):
    """True when `bin_edges` is a category order rather than numeric edges.

    Everything downstream of `cut_by_abs` takes `bin_edges`, so one test here
    lets a caller pass an already-categorical column's levels — a consequence
    class or an exon position has no numeric edges to cut.
    """
    return len(bin_edges) > 0 and isinstance(bin_edges[0], str)


def bin_labels(edges, include_nan_bin=False):
    """String labels matching the bin keys every function here returns."""
    if _is_categorical_bins(edges):
        out = list(edges)
    else:
        out = [f"{edges[i]:,g}-{edges[i+1]:,g}" for i in range(len(edges) - 1)]
    if include_nan_bin:
        out.append("NaN")
    return out


def cut_by_abs(values, bin_edges, include_nan_bin=False):
    """pd.cut(|values|, bins=bin_edges) with the standard '{a}-{b}' labels.

    When include_nan_bin is True, rows with values outside the edges (or NaN)
    are placed in an extra 'NaN' category. Returns (binned Series, labels)
    so callers can assign to a column and aggregate by bin. Pass a pandas
    Series to preserve the original index; arrays get a default range index.

    `bin_edges` may instead be the category order of an already-categorical
    column, in which case the values pass through unchanged (no absolute
    value, no cut) as an ordered Categorical over those levels.
    """
    if _is_categorical_bins(bin_edges):
        labels = list(bin_edges)
        binned = pd.Series(values).astype(pd.CategoricalDtype(labels, ordered=True))
    else:
        edges = list(bin_edges)
        labels = [f"{edges[i]:,g}-{edges[i+1]:,g}" for i in range(len(edges) - 1)]
        if isinstance(values, pd.Series):
            binned = pd.cut(
                values.abs(), bins=edges, labels=labels, include_lowest=True
            )
        else:
            binned = pd.Series(
                pd.cut(np.abs(values), bins=edges, labels=labels, include_lowest=True)
            )
    if include_nan_bin:
        binned = binned.cat.add_categories(["NaN"]).fillna("NaN")
    return binned, labels


def _too_few_clf(y, min_n):
    """True if a classification cell has fewer than min_n of either class."""
    n_pos = int(np.sum(y))
    return n_pos < min_n or (len(y) - n_pos) < min_n


################################################################################
# the bootstrap primitive
################################################################################
def boot_pvalue(boots, n_boot=N_BOOT, axis=0):
    """Two-sided percentile p: twice the smaller tail about zero.

    One convention for every variant bootstrap here. Both tails count the
    resamples landing exactly on zero, since dropping the ties would understate
    p on exactly the strata that are least resolved; on a discrete statistic
    (rho over a handful of points) the two tails can then sum past one, which
    the clip absorbs along with the 1/n_boot floor. Non-finite resamples are
    excluded from both the tails and the denominator, and a position with none
    finite returns NaN.
    """
    boots = np.asarray(boots, float)
    fin = np.isfinite(boots)
    n = fin.sum(axis=axis)
    tail = np.minimum(
        ((boots <= 0) & fin).sum(axis=axis), ((boots >= 0) & fin).sum(axis=axis)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.clip(2 * tail / n, 1 / n_boot, 1)
    p = np.where(n == 0, np.nan, p)
    return p if p.ndim else float(p)


def cluster_index(variant):
    """Group rows by variant. Returns (order, start, size, code).

    `order` sorts rows by variant; variant `c` owns `order[start[c]:][:size[c]]`.
    `code` is each row's variant, coded to 0..nv-1 in sorted variant order — the
    space strata are expressed in.
    """
    _, code = np.unique(variant, return_inverse=True)
    nv = int(code.max()) + 1
    order = np.argsort(code, kind="stable")
    size = np.bincount(code, minlength=nv)
    start = np.r_[0, np.cumsum(size)[:-1]]
    return order, start, size, code


def _gather(order, start, size, drawn):
    """Row indices of the drawn variants, each bringing all of its rows."""
    n = size[drawn]
    off = np.repeat(start[drawn] - np.r_[0, np.cumsum(n)[:-1]], n)
    return order[off + np.arange(n.sum())]


def variant_bootstrap(index, strata, statistic, *, n_boot=N_BOOT, seed=0):
    """Resample variants within strata; evaluate `statistic` on each draw.

    index: (order, start, size, code) from `cluster_index`.
    strata: list of variant-code arrays, resampled to their own size so each
        stratum's variant count is held fixed. Positives and negatives are
        always separate strata, since a resample that changed their ratio would
        change the metric for reasons that have nothing to do with the models.
    statistic: row indices -> scalar or array. Both models must be scored inside
        it off the same indices, or the comparison stops being paired.

    Returns (point, boots), where `point` is the statistic over every row and
    `boots` stacks the resamples along axis 0.
    """
    order, start, size, _ = index
    point = np.asarray(statistic(np.arange(int(size.sum()))), dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot,) + point.shape)
    for k in range(n_boot):
        drawn = np.concatenate([rng.choice(g, g.size, replace=True) for g in strata])
        boots[k] = statistic(_gather(order, start, size, drawn))
    return point, boots


################################################################################
# statistic 1: per-tissue cells, aggregated across tissues
################################################################################
def _cluster_rows(pools, col, edges, include_nan_bin, kind, cor_col):
    """Flatten two aligned pools into parallel arrays, one row per (tissue, variant).

    Returns (variant, cell, y, s1, s2, c1, c2), where `cell` codes the
    (tissue, bin) pair as tissue*n_bins + bin and the trailing pair is None
    unless kind='cor'. Rows either model scores NaN are dropped from both, so
    the two arrays stay paired; the per-tissue functions drop per model, which
    is the one place this path deliberately differs from them.
    """
    pool1, pool2 = pools
    nb = len(bin_labels(edges, include_nan_bin=include_nan_bin))
    signed = kind == "cor"
    need = ["pred"] + ([] if include_nan_bin else [col]) + ([cor_col] if signed else [])
    parts = []
    for ti, t in enumerate(pool1):
        d1 = pool1[t]
        d2 = pool2[t].reindex(d1.index)
        if signed:
            m = (d1["label"] == "pos").to_numpy()
            d1, d2 = d1[m], d2[m]
        ok = (d1[need].notna().all(axis=1) & d2[need].notna().all(axis=1)).to_numpy()
        d1, d2 = d1[ok], d2[ok]
        code = (
            cut_by_abs(d1[col], edges, include_nan_bin)[0]
            .cat.codes.to_numpy()
            .astype(np.int64)
        )
        keep = code >= 0

        def pull(d):
            return (d["pred"] if signed else d["pred"].abs()).to_numpy()[keep]

        parts.append(
            (
                d1.index.to_numpy()[keep],
                ti * nb + code[keep],
                (d1["label"].to_numpy()[keep] == "pos").astype(np.int8),
                pull(d1),
                pull(d2),
                d1[cor_col].to_numpy()[keep] if signed else None,
                d2[cor_col].to_numpy()[keep] if signed else None,
            )
        )
    return [
        None if parts[0][i] is None else np.concatenate([p[i] for p in parts])
        for i in range(7)
    ]


def cluster_stats(
    pools,
    col,
    edges,
    *,
    kind="clf",
    cor_col="coef",
    include_nan_bin=False,
    min_n=MIN_POS,
    agg="median",
    n_boot=N_BOOT,
    seed=0,
):
    """Cluster bootstrap over variants, scoring per-tissue cells inside each draw.

    pools: (pool1, pool2), each dict[tissue -> DataFrame] from `load_qtl_pools`.
    col/edges: the attribute to stratify on and its bin edges, or the category
        order of an already-categorical column (see `cut_by_abs`).
    kind: 'clf' scores |pred| against the matched negatives, returning AUROC and
        AUPRC; 'cor' scores signed pred against `cor_col` over the positives,
        returning Spearman rho.
    agg: how each draw's per-tissue values combine — 'median' or 'mean'.

    Positives and negatives are resampled with replacement within each bin,
    holding that bin's variant count fixed, and every tissue row of a drawn
    variant travels with it; both models see the same resample, so the
    comparison stays paired. `delta` is the aggregate across tissues of the
    per-tissue difference (model 2 - model 1), and p is twice the smaller tail
    of the resampled aggregates about zero, floored at 1/n_boot.

    The (tissue, bin) cells are fixed at those passing `min_n` on the full data;
    a resample leaving one single-class scores NaN and drops out of that bin's
    aggregate.

    Returns a DataFrame of ['delta', 'p'] indexed by ('bin', 'metric') when
    kind='clf' and by 'bin' when kind='cor', with the raw draws on
    .attrs['boots'] for a caller wanting percentile CIs.
    """
    agg_fn = {"median": np.nanmedian, "mean": np.nanmean}[agg]
    bin_order = bin_labels(edges, include_nan_bin=include_nan_bin)
    nb = len(bin_order)
    var, cell, y, s1, s2, c1, c2 = _cluster_rows(
        pools, col, edges, include_nan_bin, kind, cor_col
    )
    metrics = ("AUROC", "AUPRC") if kind == "clf" else ("rho",)

    index = cluster_index(var)
    order, start, size, vcode = index
    nv = size.size

    # each variant carries one class and one bin, read off its first row
    vpos = np.zeros(nv, bool)
    vpos[vcode[y == 1]] = True
    vbin = np.empty(nv, np.int64)
    vbin[vcode[::-1]] = (cell % nb)[::-1]
    strata = [
        g
        for b in range(nb)
        for g in (
            np.flatnonzero((vbin == b) & vpos),
            np.flatnonzero((vbin == b) & ~vpos),
        )
        if g.size
    ]

    # (tissue, bin) cells scored on the full data, reused by every resample
    n_row = np.bincount(cell)
    n_pos = np.bincount(cell[y == 1], minlength=n_row.size)
    cells = np.flatnonzero(
        (n_pos >= min_n) & (n_row - n_pos >= min_n) if kind == "clf" else n_row >= min_n
    )

    def bin_aggregates(idx):
        c = cell[idx]
        o = np.argsort(c, kind="stable")
        c, idx = c[o], idx[o]
        lo = np.searchsorted(c, cells, "left")
        hi = np.searchsorted(c, cells, "right")
        acc = [[[] for _ in range(nb)] for _ in metrics]
        for c0, i0, i1 in zip(cells, lo, hi):
            r = idx[i0:i1]
            b = c0 % nb
            if kind == "clf":
                yy = y[r]
                m1, m2 = clf_metrics(yy, s1[r]), clf_metrics(yy, s2[r])
                for mi in (0, 1):
                    acc[mi][b].append(m2[mi] - m1[mi])
            else:
                acc[0][b].append(spearman(s2[r], c2[r]) - spearman(s1[r], c1[r]))
        return np.array(
            [
                [agg_fn(v) if np.any(np.isfinite(v)) else np.nan for v in row]
                for row in acc
            ]
        )

    point, boots = variant_bootstrap(
        index, strata, bin_aggregates, n_boot=n_boot, seed=seed
    )
    p = boot_pvalue(boots, n_boot)

    out, draws = {}, {}
    for mi, met in enumerate(metrics):
        for bi, b in enumerate(bin_order):
            key = (b, met) if kind == "clf" else b
            out[key] = (point[mi, bi], p[mi, bi])
            draws[key] = boots[:, mi, bi]
    res = pd.DataFrame(out, index=["delta", "p"]).T
    res.index = (
        pd.MultiIndex.from_tuples(res.index, names=["bin", "metric"])
        if kind == "clf"
        else pd.Index(list(res.index), name="bin")
    )
    res.attrs["boots"] = draws
    return res


################################################################################
# statistic 2: one row per variant, collapsed over tissues
################################################################################
def _long(pool, select):
    """Every (variant, tissue) row of a stratum, stacked across tissues."""
    return pd.concat(
        [
            select(df)
            .assign(tissue=t)
            .reset_index()[["variant", "tissue", "pred", "coef", "label"]]
            for t, df in pool.items()
        ],
        ignore_index=True,
    )


def _collapse(pool, select, pick=None):
    """One row per unique variant, aggregated over the tissues it appears in.

    `pick` maps variant to a single tissue to keep instead of averaging, which
    holds per-variant noise constant across strata that differ in how many
    tissues share their variants. Both models must be handed the same map or
    the comparison stops being paired.
    """
    long = _long(pool, select)
    both = long.groupby("variant")["label"].nunique()
    long = long[long.variant.map(both) == 1]  # never both pos and neg
    if pick is not None:
        long = long[long.tissue == long.variant.map(pick)]
    return long.groupby("variant").agg(
        pred=("pred", "mean"), coef=("coef", "mean"), label=("label", "first")
    )


def one_tissue_pick(pool, seed=0):
    """variant -> one randomly chosen tissue it appears in.

    Averaging a variant over m tissues shrinks its noise by about sqrt(m), so a
    stratification ordered by m can manufacture a trend. Passing this map to
    `collapse_stats` holds that noise flat, as the control.
    """
    long = _long(pool, lambda df: df)
    return (
        long.sample(frac=1, random_state=seed)
        .groupby("variant")
        .tissue.first()
        .to_dict()
    )


def collapse_stats(
    pools, select=None, *, metric="auprc", pick=None, n_boot=N_BOOT, seed=0
):
    """Bootstrap a metric over variants collapsed to one row each.

    pools: (pool1, pool2), each dict[tissue -> DataFrame] from `load_qtl_pools`.
    select: DataFrame -> DataFrame picking the stratum out of one tissue's table;
        defaults to the whole table.
    metric: 'auprc' or 'auroc' over |pred| against the matched negatives, or
        'spearman' between signed pred and `coef` over the positives alone.
    pick: optional variant -> tissue map from `one_tissue_pick`.

    Tissue is not the unit of replication once each variant enters once, so the
    interval resamples variants — positives and negatives separately for the
    classification metrics — and the p-value is the two-sided bootstrap
    proportion crossing zero. Prefer this to `cluster_stats` where a stratum is
    too thin to support per-tissue cells.

    Returns a dict of m1, m2, delta, lo, hi (2.5/97.5 percentiles), p, n_pos,
    n_neg.
    """
    if select is None:

        def select(df):
            return df

    a, c = (_collapse(p, select, pick) for p in pools)
    common = a.index.intersection(c.index)
    a, c = a.loc[common], c.loc[common]

    # The collapsed arrays hold every variant at once, large enough that scipy's
    # C rankdata beats the numpy kernel the cluster path uses on its small cells.
    def rho(x, y):
        return float(spearmanr(x, y).statistic)

    if metric == "spearman":
        ok = (
            (a.label == "pos")
            & np.isfinite(a.pred)
            & np.isfinite(c.pred)
            & np.isfinite(a.coef)
        ).to_numpy()
        s1, s2 = a.pred.to_numpy()[ok], c.pred.to_numpy()[ok]
        target = a.coef.to_numpy()[ok]
        variant = a.index.to_numpy()[ok]
        index = cluster_index(variant)
        strata = [np.arange(index[2].size)]

        def statistic(i):
            return rho(s2[i], target[i]) - rho(s1[i], target[i])

        n_pos, n_neg = int(target.size), np.nan
    else:
        fn = _METRIC_FNS[metric]
        y = (a.label == "pos").to_numpy().astype(int)
        s1, s2 = a.pred.abs().to_numpy(), c.pred.abs().to_numpy()
        keep = np.isfinite(s1) & np.isfinite(s2)
        y, s1, s2 = y[keep], s1[keep], s2[keep]
        variant = a.index.to_numpy()[keep]
        index = cluster_index(variant)
        vpos = np.zeros(index[2].size, bool)
        vpos[index[3][y == 1]] = True
        strata = [np.flatnonzero(vpos), np.flatnonzero(~vpos)]

        def statistic(i):
            return fn(y[i], s2[i]) - fn(y[i], s1[i])

        n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    delta, boots = variant_bootstrap(index, strata, statistic, n_boot=n_boot, seed=seed)
    finite = boots[np.isfinite(boots)]
    if metric == "spearman":
        m1, m2 = rho(s1, target), rho(s2, target)
    else:
        m1, m2 = fn(y, s1), fn(y, s2)
    return {
        "m1": m1,
        "m2": m2,
        "delta": float(delta),
        "lo": np.quantile(finite, 0.025),
        "hi": np.quantile(finite, 0.975),
        "p": boot_pvalue(boots, n_boot),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


################################################################################
# per-tissue point estimates
################################################################################
def per_tissue_metric(
    df, metric="auroc", score_col="pred", label_col="label", use_abs=True
):
    """Single-tissue AUROC or AUPRC."""
    fn = _METRIC_FNS[metric]
    work = df.dropna(subset=[score_col])
    y = (work[label_col] == "pos").astype(int).values
    s = (np.abs(work[score_col]) if use_abs else work[score_col]).values
    return fn(y, s)


def per_tissue_spearman(
    df, x_col="pred", y_col="coef", label_col="label", min_n=MIN_POS
):
    """Spearman rho between signed prediction and effect size on positives."""
    work = df[df[label_col] == "pos"].dropna(subset=[x_col, y_col])
    if len(work) < min_n:
        return np.nan
    rho, _ = spearmanr(work[x_col].values, work[y_col].values)
    return float(rho)


def per_tissue_metric_by_bin(
    df,
    bin_col,
    bin_edges,
    metric="auroc",
    score_col="pred",
    label_col="label",
    use_abs=True,
    include_nan_bin=False,
    min_n=MIN_POS,
):
    """AUROC/AUPRC within each |bin_col| bin for one tissue — no bootstrap.

    The point estimate per (tissue, bin): what a violin plots, what the paired
    across-tissue significance test consumes, and what a median table reports.
    Returns Series[bin_label -> float], NaN for degenerate bins, with the
    dropped and present bin labels stashed on .attrs so callers can aggregate
    dropouts across tissues.
    """
    fn = _METRIC_FNS[metric]
    drop_subset = [score_col] if include_nan_bin else [bin_col, score_col]
    work = df.dropna(subset=drop_subset).copy()
    work["_y"] = (work[label_col] == "pos").astype(int)
    work["_s"] = np.abs(work[score_col]) if use_abs else work[score_col]
    work["_bin"], labels = cut_by_abs(work[bin_col], bin_edges, include_nan_bin)
    out = {b: float("nan") for b in (labels + (["NaN"] if include_nan_bin else []))}
    dropped_bins, present_bins = [], []
    for b, g in work.groupby("_bin", observed=True):
        if len(g) == 0:
            continue
        present_bins.append(str(b))
        if _too_few_clf(g["_y"].values, min_n):
            dropped_bins.append(str(b))
            continue
        out[str(b)] = fn(g["_y"].values, g["_s"].values)
    res = pd.Series(out, dtype=float)
    res.attrs["dropped_bins"] = dropped_bins
    res.attrs["present_bins"] = present_bins
    return res


def per_tissue_spearman_by_bin(
    df,
    bin_col,
    bin_edges,
    x_col="pred",
    y_col="coef",
    label_col="label",
    include_nan_bin=False,
    min_n=MIN_POS,
):
    """Spearman rho on positives within each |bin_col| bin for one tissue.

    Returns Series[bin_label -> float], NaN for degenerate bins, with the
    dropped and present bin labels stashed on .attrs.
    """
    drop_subset = [x_col, y_col] if include_nan_bin else [bin_col, x_col, y_col]
    work = df[df[label_col] == "pos"].dropna(subset=drop_subset).copy()
    work["_bin"], labels = cut_by_abs(work[bin_col], bin_edges, include_nan_bin)
    out = {b: float("nan") for b in (labels + (["NaN"] if include_nan_bin else []))}
    dropped_bins, present_bins = [], []
    for b, g in work.groupby("_bin", observed=True):
        if len(g) == 0:
            continue
        present_bins.append(str(b))
        x, y = g[x_col].values, g[y_col].values
        if len(g) < min_n or np.unique(x).size < 2 or np.unique(y).size < 2:
            dropped_bins.append(str(b))
            continue
        out[str(b)] = float(spearmanr(x, y).statistic)
    res = pd.Series(out, dtype=float)
    res.attrs["dropped_bins"] = dropped_bins
    res.attrs["present_bins"] = present_bins
    return res


################################################################################
# across-tissue alternative, kept as the robustness comparison
################################################################################
def wilcoxon_line(v1, v2):
    """Run paired Wilcoxon on (v2 - v1); return (n, median_delta, W, p) or None."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    mask = np.isfinite(v1) & np.isfinite(v2)
    if mask.sum() < 1:
        return None
    d = v2[mask] - v1[mask]
    if np.all(d == 0):
        return (int(mask.sum()), 0.0, float("nan"), float("nan"))
    try:
        res = wilcoxon(v2[mask], v1[mask])
    except ValueError:
        return None
    return (
        int(mask.sum()),
        float(np.median(d)),
        float(res.statistic),
        float(res.pvalue),
    )


def paired_wilcoxon_stats(pt_df, label1, label2, group_cols):
    """Paired Wilcoxon signed-rank (label2 - label1) per group, across tissues.

    `pt_df` holds one point estimate per (tissue, group, model) — no bootstrap.
    It treats the tissue as the unit of replication, which the across-tissue ICC
    says it is not; keep it as the robustness comparison against `cluster_stats`,
    not as the primary test.

    Returns a DataFrame indexed by group_cols with columns ['delta', 'p'],
    where delta is the median per-tissue difference (label2 - label1).
    """
    key = group_cols + ["tissue"]
    a = pt_df[pt_df.model == label1][key + ["value"]]
    b = pt_df[pt_df.model == label2][key + ["value"]]
    m = a.merge(b, on=key, suffixes=("_1", "_2"))
    out = {}
    for gkey, gg in m.groupby(
        group_cols if len(group_cols) > 1 else group_cols[0], observed=True
    ):
        stats = wilcoxon_line(gg["value_1"].values, gg["value_2"].values)
        if stats is None:
            out[gkey] = (np.nan, np.nan)
        else:
            _, med, _, p = stats  # n, median(d), W, p
            out[gkey] = (float(med), float(p))
    res = pd.DataFrame(out, index=["delta", "p"]).T
    res.index = (
        pd.MultiIndex.from_tuples(res.index, names=group_cols)
        if len(group_cols) > 1
        else pd.Index(list(res.index), name=group_cols[0])
    )
    return res
