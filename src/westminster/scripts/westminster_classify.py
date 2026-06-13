#!/usr/bin/env python
from optparse import OptionParser
import joblib
import os
import json
from tqdm import tqdm
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
import xgboost as xgb

"""
westminster_classify.py

Helper script to compute classifier accuracy.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <sadp_file> <sadn_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-a",
        dest="abs_value",
        default=False,
        action="store_true",
        help="Take features absolute value [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="num_folds",
        default=8,
        type="int",
        help="Cross-validation folds [Default: %default]",
    )
    parser.add_option(
        "-i",
        dest="iterations",
        default=1,
        type="int",
        help="Cross-validation iterations [Default: %default]",
    )
    parser.add_option(
        "--indel",
        dest="indel",
        default=False,
        action="store_true",
        help="Add indel size as feature [Default: %default]",
    )
    parser.add_option(
        "--iscale",
        dest="indel_scale",
        default=1,
        type="float",
        help="Scale indel scores [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="learning_rate",
        default=0.03,
        type="float",
        help="XGBoost learning rate [Default: %default]",
    )
    parser.add_option("-m", dest="model_pkl", help="Dimension reduction model")
    parser.add_option(
        "--md",
        dest="max_depth",
        default=6,
        type="int",
        help="XGBoost max_depth [Default: %default]",
    )
    parser.add_option(
        "--msl",
        dest="msl",
        default=1,
        type="int",
        help="RandomForest min_samples_leaf [Default: %default]",
    )
    parser.add_option(
        "-n",
        dest="n_estimators",
        default=128,
        type="int",
        help="RandomForest / XGBoost n_estimators [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="class_out",
        help="Output directory [Default: %default]",
    )
    parser.add_option("-r", dest="random_seed", default=44, type="int")
    parser.add_option(
        "-s",
        dest="save_preds",
        default=False,
        action="store_true",
        help="Save predictions across iterations [Default: %default]",
    )
    parser.add_option(
        "--stat",
        dest="score_keys",
        default="logD2",
        help="HDF5 key stat to consider. [Default: %default]",
    )
    parser.add_option(
        "--gene_agg",
        dest="gene_agg",
        default="max",
        type="choice",
        choices=["max", "sum"],
        help="Reduction for collapsing pair-indexed (covgene/, gene/) scores "
        "across genes to one row per SNP; ignored for SNP-indexed cov/ stats. "
        "[Default: %default]",
    )
    parser.add_option(
        "-x",
        dest="xgboost",
        default=False,
        action="store_true",
        help="Use XGBoost [Default: %default]",
    )
    parser.add_option(
        "--lgbm",
        dest="lgbm",
        default=False,
        action="store_true",
        help="Use LightGBM [Default: %default]",
    )
    parser.add_option("-t", dest="targets_file", default=None)
    (options, args) = parser.parse_args()

    if options.xgboost and options.lgbm:
        parser.error("Options --xgboost and --lgbm are mutually exclusive.")

    if len(args) != 2:
        parser.error("Must provide positive and negative variant predictions.")
    else:
        sadp_file = args[0]
        sadn_file = args[1]

    np.random.seed(options.random_seed)

    # convert comma-separated keys to list
    options.score_keys = options.score_keys.split(",")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # read dimension reduction model
    if options.model_pkl:
        model = joblib.load(options.model_pkl)

    # slice targets
    if options.targets_file is None:
        target_slice = None
    else:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)
        target_slice = targets_df.index

    # read positive/negative variants (abs + gene aggregation applied within)
    Xp = read_scores(
        sadp_file, options.score_keys, target_slice, options.abs_value, options.gene_agg
    )
    Xn = read_scores(
        sadn_file, options.score_keys, target_slice, options.abs_value, options.gene_agg
    )

    # transformations
    if options.model_pkl:
        Xp = model.transform(Xp)
        Xn = model.transform(Xn)

    # indels
    if options.indel:
        Ip = read_indel(sadp_file)
        In = read_indel(sadn_file)
        Ip = np.expand_dims(Ip, axis=-1)
        In = np.expand_dims(In, axis=-1)
        Xp = np.concatenate([Xp, Ip], axis=1)
        Xn = np.concatenate([Xn, In], axis=1)
    elif options.indel_scale != 1:
        Ip = read_indel(sadp_file, indel_bool=True)
        In = read_indel(sadn_file, indel_bool=True)
        Xp[Ip] = options.indel_scale * Xp[Ip]
        Xn[Ip] = options.indel_scale * Xn[Ip]

    # combine
    X = np.concatenate([Xp, Xn], axis=0)
    y = np.array([True] * Xp.shape[0] + [False] * Xn.shape[0], dtype="bool")
    del Xp, Xn

    # train classifier
    if X.shape[1] == 1:
        aurocs, auprcs, fpr_folds, tpr_folds, fpr_mean, tpr_mean = folds_single(
            X, y, folds=8
        )

        # save preds
        if options.save_preds:
            np.save(f"{options.out_dir}/preds.npy", X)
    else:
        if options.xgboost:
            aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds = folds_xgboost(
                X,
                y,
                folds=options.num_folds,
                iterations=options.iterations,
                n_estimators=options.n_estimators,
                max_depth=options.max_depth,
                learning_rate=options.learning_rate,
                random_state=options.random_seed,
            )
        elif options.lgbm:
            aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds = folds_lgbm(
                X,
                y,
                folds=options.num_folds,
                iterations=options.iterations,
                n_estimators=options.n_estimators,
                max_depth=options.max_depth,
                learning_rate=options.learning_rate,
                random_state=options.random_seed,
            )
        else:
            aurocs, auprcs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds = (
                folds_randfor(
                    X,
                    y,
                    folds=options.num_folds,
                    iterations=options.iterations,
                    n_estimators=options.n_estimators,
                    min_samples_leaf=options.msl,
                    random_state=options.random_seed,
                )
            )

        # save preds
        if options.save_preds:
            np.save(f"{options.out_dir}/preds.npy", preds)

        # save full model
        if options.xgboost:
            model = full_xgboost(
                X,
                y,
                n_estimators=options.n_estimators,
                max_depth=options.max_depth,
                learning_rate=options.learning_rate,
            )
        elif options.lgbm:
            model = full_lgbm(
                X,
                y,
                n_estimators=options.n_estimators,
                max_depth=options.max_depth,
                learning_rate=options.learning_rate,
            )
        else:
            model = full_randfor(
                X, y, n_estimators=options.n_estimators, min_samples_leaf=options.msl
            )
        joblib.dump(model, f"{options.out_dir}/model.pkl")

        # save model hyperparameters for reproducibility
        model_params = model.get_params() if hasattr(model, "get_params") else {}
        meta = {
            "classifier": (
                "xgboost"
                if options.xgboost
                else "lightgbm" if options.lgbm else "random_forest"
            ),
            "model_params": model_params,
            "folds": options.num_folds,
            "iterations": options.iterations,
            "score_keys": options.score_keys,
            "gene_agg": options.gene_agg,
            "abs_value": options.abs_value,
            "indel": options.indel,
            "indel_scale": options.indel_scale,
            "targets_file": options.targets_file,
            "random_seed": options.random_seed,
        }
        with open(os.path.join(options.out_dir, "model_params.json"), "w") as jf:
            json.dump(meta, jf, indent=2)

    # save
    np.save(f"{options.out_dir}/aurocs.npy", aurocs)
    np.save(f"{options.out_dir}/fpr_mean.npy", fpr_mean)
    np.save(f"{options.out_dir}/tpr_mean.npy", tpr_mean)

    # print stats
    with open(f"{options.out_dir}/stats.txt", "w") as stats_out:
        auroc_stdev = np.std(aurocs) / np.sqrt(len(aurocs))
        print(f"AUROC: {np.mean(aurocs):.4f} ({auroc_stdev:.4f})", file=stats_out)

    # plot roc
    plot_roc(fpr_folds, tpr_folds, options.out_dir)


def folds_randfor(
    X: np.array,
    y: np.array,
    folds: int = 8,
    iterations: int = 1,
    n_estimators: int = 100,
    min_samples_leaf: int = 1,
    random_state: int = 44,
):
    """
    Fit random forest classifiers in cross-validation.

    Args:
      X (:obj:`np.array`):
        Feature matrix.
      y (:obj:`np.array`):
        Target values.
      folds (:obj:`int`):
        Cross folds.
      iterations (:obj:`int`):
        Cross fold iterations.
      n_estimators (:obj:`int`):
        Number of trees in the forest.
      min_samples_leaf (:obj:`float`):
        Minimum number of samples required to be at a leaf node.
      random_state (:obj:`int`):
        sklearn random_state.
    """
    aurocs = []
    auprcs = []
    fpr_folds = []
    tpr_folds = []
    fpr_fulls = []
    tpr_fulls = []
    preds_return = []

    fpr_mean = np.linspace(0, 1, 256)
    tpr_mean = []

    for i in tqdm(range(iterations)):
        rs_iter = random_state + i
        preds_full = np.zeros(y.shape)

        kf = KFold(n_splits=folds, shuffle=True, random_state=rs_iter)

        for train_index, test_index in kf.split(X):
            # fit model
            if random_state is None:
                rs_rf = None
            else:
                rs_rf = rs_iter + test_index[0]
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="log2",
                min_samples_leaf=min_samples_leaf,
                random_state=rs_rf,
                n_jobs=-1,
            )
            model.fit(X[train_index, :], y[train_index])

            # predict test set
            preds = model.predict_proba(X[test_index, :])[:, 1]

            # save
            preds_full[test_index] = preds.squeeze()

            # compute ROC curve
            fpr, tpr, _ = roc_curve(y[test_index], preds)
            fpr_folds.append(fpr)
            tpr_folds.append(tpr)

            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_mean.append(interp_tpr)

            # compute AUROC and AUPRC
            aurocs.append(roc_auc_score(y[test_index], preds))
            auprcs.append(average_precision_score(y[test_index], preds))

        fpr_full, tpr_full, _ = roc_curve(y, preds_full)
        fpr_fulls.append(fpr_full)
        tpr_fulls.append(tpr_full)
        preds_return.append(preds_full)

    aurocs = np.array(aurocs)
    auprcs = np.array(auprcs)
    tpr_mean = np.array(tpr_mean).mean(axis=0)
    preds_return = np.array(preds_return).T

    return aurocs, auprcs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds_return


def folds_single(X: np.array, y: np.array, folds: int = 8, random_state: int = 44):
    """
    Compute ROC for a single value, sans model.

    Args:
      X (:obj:`np.array`):
        Feature values.
      y (:obj:`np.array`):
        Target values.
      folds (:obj:`int`):
        Cross folds.
      random_state (:obj:`int`):
        sklearn random_state.
    """
    aurocs = []
    auprcs = []
    fpr_folds = []
    tpr_folds = []

    fpr_mean = np.linspace(0, 1, 256)
    tpr_mean = []

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X):
        # predict test set (as is)
        preds = X[test_index, :]

        # compute ROC curve
        fpr, tpr, _ = roc_curve(y[test_index], preds)
        fpr_folds.append(fpr)
        tpr_folds.append(tpr)

        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_mean.append(interp_tpr)

        # compute AUROC and AUPRC
        aurocs.append(roc_auc_score(y[test_index], preds))
        auprcs.append(average_precision_score(y[test_index], preds))

    tpr_mean = np.array(tpr_mean).mean(axis=0)

    return (
        np.array(aurocs),
        np.array(auprcs),
        np.array(fpr_folds),
        np.array(tpr_folds),
        fpr_mean,
        tpr_mean,
    )


def folds_xgboost(
    X: np.array,
    y: np.array,
    folds: int = 8,
    iterations: int = 1,
    n_estimators: int = 32,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 44,
):
    """
    Fit random forest classifiers in cross-validation.

    Args:
      X (:obj:`np.array`):
        Feature matrix.
      y (:obj:`np.array`):
        Target values.
      folds (:obj:`int`):
        Cross folds.
      iterations (:obj:`int`):
        Cross fold iterations.
      n_estimators (:obj:`int`):
        Number of trees in the forest.
      max_depth (:obj:`int`):
        Maximum tree depth.
      learning_rate (:obj:`float`):
        Boosting learning rate.
      random_state (:obj:`int`):
        sklearn random_state.
    """
    aurocs = []
    fpr_folds = []
    tpr_folds = []
    fpr_fulls = []
    tpr_fulls = []
    preds_return = []

    fpr_mean = np.linspace(0, 1, 256)
    tpr_mean = []

    for i in tqdm(range(iterations)):
        rs_iter = random_state + i
        preds_full = np.zeros(y.shape)

        kf = KFold(n_splits=folds, shuffle=True, random_state=rs_iter)

        for train_index, test_index in kf.split(X):
            # fit model
            if random_state is None:
                rs_rf = None
            else:
                rs_rf = rs_iter + test_index[0]

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                colsample_bytree=0.5,
                min_child_weight=2,
                random_state=rs_rf,
            )
            # colsample_bytree=1/np.sqrt(X.shape[1]),
            # min_child_weight=[1,2],
            # subsample=1,
            # reg_alpha=1,

            model.fit(X[train_index, :], y[train_index])

            # predict test set
            preds = model.predict_proba(X[test_index, :])[:, 1]

            # save
            preds_full[test_index] = preds.squeeze()

            # compute ROC curve
            fpr, tpr, _ = roc_curve(y[test_index], preds)
            fpr_folds.append(fpr)
            tpr_folds.append(tpr)

            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_mean.append(interp_tpr)

            # compute AUROC
            aurocs.append(roc_auc_score(y[test_index], preds))

        fpr_full, tpr_full, _ = roc_curve(y, preds_full)
        fpr_fulls.append(fpr_full)
        tpr_fulls.append(tpr_full)
        preds_return.append(preds_full)

    aurocs = np.array(aurocs)
    tpr_mean = np.array(tpr_mean).mean(axis=0)
    preds_return = np.array(preds_return).T

    return aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds_return


def full_randfor(
    X: np.array,
    y: np.array,
    n_estimators: int = 100,
    min_samples_leaf: int = 1,
    random_state: int = 44,
):
    """
    Fit a single random forest on the full data.

    Args:
      X (:obj:`np.array`):
        Feature matrix.
      y (:obj:`np.array`):
        Target values.
      n_estimators (:obj:`int`):
        Number of trees in the forest.
      min_samples_leaf (:obj:`float`):
        Minimum number of samples required to be at a leaf node.
      random_state (:obj:`int`):
        sklearn random_state.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="log2",
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def full_xgboost(
    X: np.array,
    y: np.array,
    n_estimators: int = 32,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 44,
):
    """
    Fit a single xgboost forest on the full data.

    Args:
      X (:obj:`np.array`):
        Feature matrix.
      y (:obj:`np.array`):
        Target values.
      n_estimators (:obj:`int`):
        Number of trees in the forest.
      max_depth (:obj:`int`):
        Maximum tree depth.
      learning_rate (:obj:`float`):
        Boosting learning rate.
      random_state (:obj:`int`):
        sklearn random_state.
    """
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        n_jobs=-1,
        colsample_bytree=0.5,
        min_child_weight=2,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def folds_lgbm(
    X: np.array,
    y: np.array,
    folds: int = 8,
    iterations: int = 1,
    n_estimators: int = 32,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 44,
):
    """
    Fit LightGBM classifiers in cross-validation.

    Args:
      X (:obj:`np.array`):
        Feature matrix.
      y (:obj:`np.array`):
        Target values.
      folds (:obj:`int`):
        Cross folds.
      iterations (:obj:`int`):
        Cross fold iterations.
      n_estimators (:obj:`int`):
        Number of boosting rounds.
      max_depth (:obj:`int`):
        Maximum tree depth.
      learning_rate (:obj:`float`):
        Boosting learning rate.
      random_state (:obj:`int`):
        Random state.
    """
    lgb = _import_lightgbm()

    aurocs = []
    fpr_folds = []
    tpr_folds = []
    fpr_fulls = []
    tpr_fulls = []
    preds_return = []

    fpr_mean = np.linspace(0, 1, 256)
    tpr_mean = []

    for i in tqdm(range(iterations)):
        rs_iter = random_state + i
        preds_full = np.zeros(y.shape)

        kf = KFold(n_splits=folds, shuffle=True, random_state=rs_iter)

        for train_index, test_index in kf.split(X):
            rs_rf = None if random_state is None else rs_iter + test_index[0]

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=rs_rf,
                verbose=-1,
            )
            model.fit(X[train_index, :], y[train_index])

            preds = model.predict_proba(X[test_index, :])[:, 1]
            preds_full[test_index] = preds.squeeze()

            fpr, tpr, _ = roc_curve(y[test_index], preds)
            fpr_folds.append(fpr)
            tpr_folds.append(tpr)

            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_mean.append(interp_tpr)

            aurocs.append(roc_auc_score(y[test_index], preds))

        fpr_full, tpr_full, _ = roc_curve(y, preds_full)
        fpr_fulls.append(fpr_full)
        tpr_fulls.append(tpr_full)
        preds_return.append(preds_full)

    aurocs = np.array(aurocs)
    tpr_mean = np.array(tpr_mean).mean(axis=0)
    preds_return = np.array(preds_return).T

    return aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds_return


def full_lgbm(
    X: np.array,
    y: np.array,
    n_estimators: int = 32,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 44,
):
    """
    Fit a single LightGBM classifier on the full data.

    Args:
      X (:obj:`np.array`):
        Feature matrix.
      y (:obj:`np.array`):
        Target values.
      n_estimators (:obj:`int`):
        Number of boosting rounds.
      max_depth (:obj:`int`):
        Maximum tree depth.
      learning_rate (:obj:`float`):
        Boosting learning rate.
      random_state (:obj:`int`):
        Random state.
    """
    lgb = _import_lightgbm()

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _import_lightgbm():
    """Import LightGBM only when needed for --lgbm workflows."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for --lgbm. Install it with `pip install lightgbm`."
        ) from exc
    return lgb


def plot_roc(fprs: np.array, tprs: np.array, out_dir: str):
    """
    Plot ROC curve.

    Args:
      fprs (:obj:`np.array`):
        False positive rates
      tprs (:obj:`np.array`):
        True positive rates.
      out_dir (:obj:`str`):
        Output directory.
    """
    plt.figure(figsize=(4, 4))

    for fi in range(len(fprs)):
        plt.plot(fprs[fi], tprs[fi], alpha=0.25)

    ax = plt.gca()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    sns.despine()
    plt.tight_layout()

    plt.savefig("%s/roc.pdf" % out_dir)
    plt.close()


def read_indel(h5_file, indel_abs=True, indel_bool=False):
    """
    Read indel allele information from HDF5 file.

    Args:
      h5_file (:obj:`str`):
        Stats HDF5 file.
      indel_abs (:obj:`bool`):
        Take absolute value of indel size.
      indel_bool (:obj:`bool`):
        Return boolean indel indication vector.
    """
    with h5py.File(h5_file, "r") as h5_open:
        try:
            ref_alleles = [ra.decode("UTF-8") for ra in h5_open["ref_allele"]]
            alt_alleles = [aa.decode("UTF-8") for aa in h5_open["alt_allele"]]
        except KeyError:
            ref_alleles = [ra.decode("UTF-8") for ra in h5_open["ref"]]
            alt_alleles = [aa.decode("UTF-8") for aa in h5_open["alt"]]
    num_variants = len(ref_alleles)
    indels = np.array(
        [len(ref_alleles[vi]) - len(alt_alleles[vi]) for vi in range(num_variants)]
    )
    if indel_abs:
        indels = np.abs(indels)
    if indel_bool:
        indels = indels != 0
    return indels


def read_scores(
    stats_h5_file: str,
    score_keys: List[str],
    target_slice: np.array,
    abs_value: bool = False,
    gene_agg: str = "max",
):
    """
    Read variant scores from HDF5 file.

    For pair-indexed stats (covgene/, gene/), the file carries a `snp_idx`
    dataset mapping each row to its base SNP, so one SNP spans several gene rows.
    Those rows are always collapsed to a single feature vector per SNP (genes
    weighted evenly) via the `gene_agg` reduction. SNP-indexed cov/ stats have
    one row per SNP and no gene structure, so `gene_agg` is a no-op for them.

    Combining stats from different index families (e.g. cov/ with covgene/) is
    supported: each pair-indexed key is aggregated to per-SNP and reindexed onto
    the full SNP set (the cov/ rows), filling SNPs with no gene rows with zeros,
    so all keys share one row per SNP before concatenation. When every key is
    pair-indexed, the compact per-SNP-with-genes ordering is kept instead.

    `abs_value` is applied before aggregation so that "max" selects the
    strongest-magnitude gene. Each pair-indexed key is reduced in its native
    HDF5 dtype (f16) and only the collapsed result is upcast to float32, so the
    full pair matrix is never materialized at float32.

    Args:
      stats_h5_file (:obj:`str`):
        Stats HDF5 file.
      score_keys (:obj:`List[str]`):
        Stat HDF5 keys.
      target_slice (:obj:`np.array`):
        Target axis slice.
      abs_value (:obj:`bool`):
        Take absolute value of features.
      gene_agg (:obj:`str`):
        Per-SNP gene aggregation: "max" or "sum".
    """
    with h5py.File(stats_h5_file, "r") as stats_h5:
        snp_idx = stats_h5["snp_idx"][:] if "snp_idx" in stats_h5 else None

        # covgene/ and gene/ stats are pair-indexed (one row per SNP x gene)
        pair_indexed = [
            snp_idx is not None
            and (sk.startswith("covgene/") or sk.startswith("gene/"))
            for sk in score_keys
        ]

        # pair-indexed rows (one per SNP x gene) are always collapsed to one row
        # per SNP; the reduction is a no-op for SNP-indexed cov/ stats
        aggregate = any(pair_indexed)
        mixed = aggregate and not all(pair_indexed)

        # contiguous SNP groups for collapsing pair rows, plus the full SNP
        # count needed to align pair-indexed keys with SNP-indexed ones
        if aggregate:
            order, group_starts, group_snps = gene_groups(snp_idx)
            n_snps = (
                next(
                    stats_h5[sk].shape[0]
                    for sk, p in zip(score_keys, pair_indexed)
                    if not p
                )
                if mixed
                else None
            )
        reduceat = np.maximum.reduceat if gene_agg == "max" else np.add.reduceat

        key_scores = []
        for sk, is_pair in zip(score_keys, pair_indexed):
            score = stats_h5[sk][:]
            if target_slice is not None:
                score = score[..., target_slice]
            np.nan_to_num(score, copy=False)
            if abs_value:
                np.abs(score, out=score)

            if aggregate and is_pair:
                # collapse gene rows in native dtype, then upcast small result
                if order is not None:
                    score = score[order]
                score = reduceat(score, group_starts, axis=0).astype("float32")
                if mixed:
                    full = np.zeros((n_snps, score.shape[1]), dtype="float32")
                    full[group_snps] = score
                    score = full
            else:
                score = score.astype("float32")

            key_scores.append(score)

    return np.concatenate(key_scores, axis=-1)


def gene_groups(snp_idx: np.array):
    """Contiguous SNP groupings for collapsing pair-indexed rows.

    Returns `(order, group_starts, group_snps)`: rows taken in `order` (None
    when `snp_idx` is already non-decreasing) fall into contiguous per-SNP
    groups beginning at `group_starts`, with `group_snps` the SNP index of each
    group. Groups are ordered by ascending SNP index, so the collapsed rows are
    deterministic and consistent between the positive and negative files.
    """
    if np.all(np.diff(snp_idx) >= 0):
        order, sorted_idx = None, snp_idx
    else:
        order = np.argsort(snp_idx, kind="stable")
        sorted_idx = snp_idx[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_idx)) + 1]
    return order, group_starts, sorted_idx[group_starts]


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
