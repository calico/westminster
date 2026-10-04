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
    replicate_stats,
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
# replicate-level comparison
################################################################################
def make_reps(n, edge=0.0, seed_sd=0.05, seed=0):
    """n replicates of one config: a shared pool plus per-replicate seed noise."""
    base = make_pools(edge=edge)[1]
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        shift = rng.normal(0, seed_sd, len(next(iter(base.values()))))
        reps.append({t: df.assign(pred=df.pred + shift) for t, df in base.items()})
    return reps


def test_replicate_stats_identical_configs():
    reps = make_reps(3)
    res, _ = replicate_stats((reps, reps), "REGION", ["TSS", "CDS"], min_n=5, n_boot=50)
    assert res["delta"].abs().max() == 0
    assert (res["p"] == 1).all()


def test_replicate_stats_detects_a_real_edge_with_unequal_replicates():
    reps1, reps2 = make_reps(4, seed=1), make_reps(2, edge=0.8, seed=2)
    res, (r1, r2) = replicate_stats((reps1, reps2), None, None, min_n=5, n_boot=100)
    assert (res["delta"] > 0).all()
    assert (res["p"] < 0.05).all()
    assert (res["se_seed"] > 0).all() and (res["se_variant"] > 0).all()
    assert r1.shape == (2, 4) and r2.shape == (2, 2)
    np.testing.assert_allclose(r2.mean(axis=1) - r1.mean(axis=1), res["delta"])


def test_replicate_stats_seed_noise_widens_the_test():
    kw = dict(min_n=5, n_boot=100)
    quiet, _ = replicate_stats((make_reps(3, seed_sd=0.01, seed=1),
                             make_reps(3, edge=0.1, seed_sd=0.01, seed=2)), None, None, **kw)
    noisy, _ = replicate_stats((make_reps(3, seed_sd=0.5, seed=1),
                             make_reps(3, edge=0.1, seed_sd=0.5, seed=2)), None, None, **kw)
    assert (noisy["se_seed"] > quiet["se_seed"]).all()


def test_replicate_stats_cor_kind_and_single_replicate():
    res, _ = replicate_stats((make_reps(1), make_reps(3, edge=0.8)), "REGION",
                          ["TSS", "CDS"], kind="cor", min_n=5, n_boot=50)
    assert list(res.index) == ["TSS", "CDS"]
    assert res["p"].isna().all() and res["se_seed"].isna().all()
    assert res["se_variant"].notna().all()


@pytest.mark.parametrize("agg", ["mean", "median"])
@pytest.mark.parametrize("shared_tissue", [False, True])
def test_replicate_stats_cor_uses_shared_valid_cells(agg, shared_tissue):
    x = np.arange(30, dtype=float)
    df = pd.DataFrame(
        {"label": "pos", "coef": x, "pred": -x},
        index=[f"v{i}" for i in range(len(x))],
    )
    a = {"constant_in_one_replicate": df}
    b = {"constant_in_one_replicate": df.assign(pred=0.0)}
    if shared_tissue:
        a["shared"] = b["shared"] = df.assign(pred=x)

    res, vals = replicate_stats(
        ([a, a], [a, b]), None, None, kind="cor", agg=agg, n_boot=30
    )
    if shared_tissue:
        assert res.loc["all", "delta"] == 0
        assert res.loc["all", "p"] == 1
        for values in vals:
            np.testing.assert_allclose(values, 1)
    else:
        assert res.isna().all().all()
        assert all(values.isna().all().all() for values in vals)


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


@pytest.mark.parametrize('kind', ['clf', 'cor'])
@pytest.mark.parametrize('shared', [False, True])
def test_weighted_cell_matches_expanded_samples(kind, shared):
    from westminster.bootstrap import _prepare_cell

    rng = np.random.default_rng(51)
    y = rng.integers(0, 2, 40)
    scores = rng.integers(-3, 4, (3, 40)).astype(float)
    effects = rng.integers(-2, 3, (3, 40)).astype(float)
    if shared:
        effects[:] = effects[0]
    evaluate = _prepare_cell(y, scores, effects if kind == 'cor' else None)
    weights = [np.zeros(40, int), np.ones(40, int), y, 1 - y,
               np.eye(1, 40, dtype=int)[0] * 3]
    weights += [rng.integers(0, 5, 40) for _ in range(20)]
    for w in weights:
        idx = np.repeat(np.arange(40), w)
        expected = np.full((3, 2 if kind == 'clf' else 1), np.nan)
        for k in range(3):
            if kind == 'clf' and np.unique(y[idx]).size == 2:
                expected[k] = [roc_auc_score(y[idx], scores[k, idx]),
                               average_precision_score(y[idx], scores[k, idx])]
            elif kind == 'cor' and len(idx) > 1 and all(
                    np.unique(a[idx]).size > 1 for a in (scores[k], effects[k])):
                expected[k, 0] = spearmanr(scores[k, idx], effects[k, idx]).statistic
        np.testing.assert_allclose(evaluate(w), expected, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize('kind', ['clf', 'cor'])
@pytest.mark.parametrize('counts', [False, True])
@pytest.mark.parametrize('col,edges,nan_bin', [(None, None, False),
    ('REGION', ['TSS', 'CDS', 'missing'], True), ('dist', [0, 1, 3], False)])
def test_prepared_cluster_matches_expanded_draws(kind, counts, col, edges, nan_bin):
    from westminster.bootstrap import _cluster_rows, _cluster_setup

    pools = make_pools(n_var=24, n_tissue=3, edge=0.3)
    for k, pool in enumerate(pools):
        for ti, (t, df) in enumerate(pool.items()):
            df['pred'] = df.pred.round(0)
            df['coef'] = (df.coef + k * df.pred).round(0)
            df['dist'] = np.arange(len(df)) % 5
            df.loc['v2', 'REGION'] = np.nan
            df.loc['v4', 'pred'] = np.nan
            pool[t] = df.iloc[ti:].iloc[::-1]
    index, strata, cells, score = _cluster_setup(
        pools, col, edges, nan_bin, kind, 'coef', 2, counts=counts)
    _, cell, y, S, C = _cluster_rows(pools, col, edges, nan_bin, kind, 'coef')

    def expanded(idx):
        out = np.full((2, 2 if kind == 'clf' else 1, len(cells)), np.nan)
        for j, c in enumerate(cells):
            rows = idx[cell[idx] == c]
            for k in range(2):
                out[k, :, j] = (clf_metrics(y[rows], S[k, rows]) if kind == 'clf'
                                else spearman(S[k, rows], C[k, rows]))
        return out

    actual = variant_bootstrap(index, strata, score, n_boot=40, seed=9, counts=counts)
    expected = variant_bootstrap(index, strata, expanded, n_boot=40, seed=9)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=1e-14, equal_nan=True)
