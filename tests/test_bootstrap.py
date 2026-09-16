from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from westminster.bootstrap import (
    boot_pvalue,
    clf_metrics,
    cluster_index,
    cluster_stats,
    collapse_stats,
    one_tissue_pick,
    spearman,
    variant_bootstrap,
)
from westminster.scripts.westminster_qtl_cmp import tissue_unit_table


################################################################################
# metric kernels
################################################################################
@pytest.mark.parametrize("n_levels", [2, 5, 50])
def test_clf_metrics_matches_sklearn(n_levels):
    """Heavy ties are the reason these are hand-rolled, so test them."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.integers(0, n_levels, 400).astype(float)
    auroc, auprc = clf_metrics(y, s)
    assert auroc == pytest.approx(roc_auc_score(y, s))
    assert auprc == pytest.approx(average_precision_score(y, s))


def test_clf_metrics_single_class():
    assert all(np.isnan(v) for v in clf_metrics(np.ones(10), np.arange(10.0)))


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 6, 200).astype(float)
    y = rng.integers(0, 4, 200).astype(float)
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_spearman_constant_is_nan():
    assert np.isnan(spearman(np.ones(10), np.arange(10.0)))


################################################################################
# boot_pvalue
################################################################################
def test_boot_pvalue_floor_and_ceiling():
    assert boot_pvalue(np.ones(100), n_boot=100) == pytest.approx(0.01)
    # every draw on zero counts in both tails, so p saturates at 1
    assert boot_pvalue(np.zeros(100), n_boot=100) == pytest.approx(1.0)


def test_boot_pvalue_two_sided():
    boots = np.r_[np.full(90, 1.0), np.full(10, -1.0)]
    assert boot_pvalue(boots, n_boot=100) == pytest.approx(0.2)


def test_boot_pvalue_ignores_nonfinite():
    boots = np.r_[np.full(45, 1.0), np.full(5, -1.0), np.full(50, np.nan)]
    assert boot_pvalue(boots, n_boot=100) == pytest.approx(0.2)
    assert np.isnan(boot_pvalue(np.full(10, np.nan), n_boot=10))


def test_boot_pvalue_axis():
    boots = np.stack([np.ones(100), -np.ones(100)], axis=1)
    assert boot_pvalue(boots, n_boot=100).tolist() == pytest.approx([0.01, 0.01])


################################################################################
# the resampler
################################################################################
def _index_of(variants):
    return cluster_index(np.array(variants))


def test_variant_bootstrap_is_deterministic():
    index = _index_of(list("abcdef"))
    strata = [np.arange(6)]

    def stat(i):
        return float(i.sum())

    _, b1 = variant_bootstrap(index, strata, stat, n_boot=20, seed=3)
    _, b2 = variant_bootstrap(index, strata, stat, n_boot=20, seed=3)
    _, b3 = variant_bootstrap(index, strata, stat, n_boot=20, seed=4)
    assert np.array_equal(b1, b2)
    assert not np.array_equal(b1, b3)


def test_variant_bootstrap_carries_whole_clusters():
    """A drawn variant brings every one of its rows, so row counts come in
    multiples of the cluster size and a variant is never split."""
    variants = np.array(["v0"] * 3 + ["v1"] * 3 + ["v2"] * 3)
    index = _index_of(variants)
    seen = []

    def stat(rows):
        counts = np.bincount(index[3][rows], minlength=3)
        seen.append(counts)
        return float(rows.size)

    point, boots = variant_bootstrap(index, [np.arange(3)], stat, n_boot=50, seed=0)
    assert point == 9  # the point estimate is every row, once
    assert np.all(boots == 9)  # 3 variants drawn, 3 rows each
    for counts in seen:
        assert set(np.unique(counts)) <= {0, 3, 6, 9}


def test_variant_bootstrap_holds_stratum_sizes_fixed():
    index = _index_of(list("abcdef"))
    strata = [np.arange(2), np.arange(2, 6)]
    drawn_pos = []

    def stat(rows):
        drawn_pos.append(int(np.sum(index[3][rows] < 2)))
        return 0.0

    variant_bootstrap(index, strata, stat, n_boot=30, seed=0)
    assert set(drawn_pos[1:]) == {2}  # always 2 from the first stratum


################################################################################
# end-to-end on a synthetic benchmark
################################################################################
def make_pools(n_var=60, n_tissue=6, edge=0.0, seed=0):
    """Two pools over the same variants; model 2 reads the effect `edge` cleaner.

    Positives carry a real effect and negatives carry none, so |pred| is
    genuinely discriminative. Both models see the same effect through the same
    noise draw, model 2 with `edge` of that noise removed. Each variant recurs
    in every tissue with nearly the same prediction, the correlation structure
    the variant bootstrap exists to handle.
    """
    rng = np.random.default_rng(seed)
    variants = [f"v{i}" for i in range(n_var)]
    label = np.where(np.arange(n_var) % 2 == 0, "pos", "neg")
    effect = np.where(label == "pos", rng.normal(0, 1, n_var), 0.0)
    coef = np.where(label == "pos", effect, np.nan)
    region = np.where(np.arange(n_var) % 4 < 2, "TSS", "CDS")
    noise = rng.normal(0, 1, n_var)  # per variant, not per tissue: high ICC

    pool1, pool2 = {}, {}
    for t in range(n_tissue):
        jitter = rng.normal(0, 0.02, n_var)
        common = dict(variant=variants, label=label, coef=coef, REGION=region)
        pool1[f"t{t}"] = pd.DataFrame(
            dict(common, pred=effect + noise + jitter)
        ).set_index("variant")
        pool2[f"t{t}"] = pd.DataFrame(
            dict(common, pred=effect + noise * (1 - edge) + jitter)
        ).set_index("variant")
    return pool1, pool2


def test_cluster_stats_identical_models():
    pool = make_pools()[0]
    res = cluster_stats((pool, pool), "REGION", ["TSS", "CDS"], min_n=5, n_boot=100)
    assert res["delta"].abs().max() == 0
    assert (res["p"] == 1).all()


def test_cluster_stats_detects_a_real_edge():
    pools = make_pools(edge=0.8)
    res = cluster_stats(pools, "REGION", ["TSS", "CDS"], min_n=5, n_boot=200)
    assert (res["delta"] > 0).all()
    assert (res["p"] == 1 / 200).all()  # no draw crosses zero: the floor
    assert list(res.index.names) == ["bin", "metric"]
    assert set(res.index.get_level_values("metric")) == {"AUROC", "AUPRC"}


def test_cluster_stats_cor_kind():
    pools = make_pools(edge=0.8)
    res = cluster_stats(
        pools, "REGION", ["TSS", "CDS"], kind="cor", min_n=5, n_boot=100
    )
    assert res.index.name == "bin"
    assert list(res.index) == ["TSS", "CDS"]
    assert set(res.attrs["boots"]) == {"TSS", "CDS"}


def test_cluster_stats_agg_mean_differs_from_median():
    pools = make_pools(edge=0.3)
    kw = dict(min_n=5, n_boot=100)
    med = cluster_stats(pools, "REGION", ["TSS", "CDS"], **kw)
    mean = cluster_stats(pools, "REGION", ["TSS", "CDS"], agg="mean", **kw)
    assert not np.allclose(med["delta"], mean["delta"])


################################################################################
# the collapsed flavor
################################################################################
def test_collapse_stats_brackets_its_delta():
    pools = make_pools(edge=0.8)
    res = collapse_stats(pools, metric="auprc", n_boot=200)
    assert res["lo"] <= res["delta"] <= res["hi"]
    assert res["delta"] == pytest.approx(res["m2"] - res["m1"])
    assert res["n_pos"] == 30 and res["n_neg"] == 30


def test_collapse_stats_identical_models():
    pool = make_pools()[0]
    res = collapse_stats((pool, pool), metric="auprc", n_boot=100)
    assert res["delta"] == 0
    assert res["p"] == pytest.approx(1.0)


def test_collapse_stats_spearman_uses_positives_only():
    pools = make_pools(edge=0.8)
    res = collapse_stats(pools, metric="spearman", n_boot=100)
    assert res["n_pos"] == 30
    assert np.isnan(res["n_neg"])


def test_collapse_stats_select_narrows_the_stratum():
    pools = make_pools(edge=0.8)
    res = collapse_stats(
        pools, lambda df: df[df.REGION == "TSS"], metric="auprc", n_boot=100
    )
    assert res["n_pos"] == 15


def test_collapse_stats_without_an_effect_size_column():
    """sQTL and paQTL tables carry no coef; classification must not need one."""
    pools = tuple(
        {t: df.drop(columns="coef") for t, df in pool.items()}
        for pool in make_pools(edge=0.8)
    )
    res = collapse_stats(pools, metric="auprc", n_boot=100)
    assert res["delta"] > 0


def test_collapse_stats_honors_cor_col():
    """Correlating against the negated column must flip the delta, not ignore it."""
    pools = tuple(
        {t: df.assign(negcoef=-df.coef) for t, df in pool.items()}
        for pool in make_pools(edge=0.8)
    )
    kw = dict(metric="spearman", n_boot=100)
    plain = collapse_stats(pools, **kw)
    flipped = collapse_stats(pools, cor_col="negcoef", **kw)
    assert flipped["delta"] == pytest.approx(-plain["delta"])
    assert flipped["m1"] == pytest.approx(-plain["m1"])


@pytest.mark.parametrize(
    "select,n_pos,n_neg",
    [
        (lambda df: df.iloc[:0], 0, 0),  # empty bin
        (lambda df: df[df.label == "pos"], 30, 0),  # one class
    ],
)
def test_collapse_stats_returns_nan_for_unscorable_strata(select, n_pos, n_neg):
    """One dead bin must cost its own row, not the whole table."""
    res = collapse_stats(make_pools(), select, metric="auprc", n_boot=50)
    assert (res["n_pos"], res["n_neg"]) == (n_pos, n_neg)
    assert all(np.isnan(res[k]) for k in ("m1", "m2", "delta", "lo", "hi", "p"))


def test_one_tissue_pick_keeps_sign():
    pools = make_pools(edge=0.8)
    pick = one_tissue_pick(pools[1])
    assert set(pick) == {f"v{i}" for i in range(60)}
    res = collapse_stats(pools, metric="auprc", pick=pick, n_boot=100)
    assert res["delta"] > 0


################################################################################
# the CLI's display layer
################################################################################
def test_tissue_unit_table_displays_the_requested_agg():
    """The model columns must combine tissues the same way the delta does."""
    pools = make_pools(edge=0.3)
    args = dict(min_n=5, n_boot=50, seed=0, cor_col="coef")
    tables = {
        agg: tissue_unit_table(
            pools,
            "REGION",
            ["TSS", "CDS"],
            "m1",
            "m2",
            Namespace(agg=agg, **args),
            True,
        )
        for agg in ("median", "mean")
    }
    assert not np.allclose(tables["median"]["m1"], tables["mean"]["m1"])
