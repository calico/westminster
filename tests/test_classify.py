#!/usr/bin/env python

import shutil
import subprocess

import h5py
import numpy as np

from westminster.scripts.westminster_classify import read_scores

"""
westminster_classify.py
"""


def test_classify():
    num_variants = 256
    out_dir = "tests/data/class_out"
    shutil.rmtree(out_dir)

    # simulate and write positive examples
    Xp = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=num_variants)
    pos_h5f = "tests/data/Xp.h5"
    with h5py.File(pos_h5f, "w") as pos_h5:
        pos_h5.create_dataset("SAD", data=Xp, dtype="float16")
        pos_h5.create_dataset("ref", data=np.array(["A"] * num_variants, dtype="S"))
        pos_h5.create_dataset("alt", data=np.array(["C"] * num_variants, dtype="S"))

    # simulate and write negative examples
    Xn = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], size=num_variants)
    neg_h5f = "tests/data/Xn.h5"
    with h5py.File(neg_h5f, "w") as neg_h5:
        neg_h5.create_dataset("SAD", data=Xn, dtype="float16")
        neg_h5.create_dataset("ref", data=np.array(["A"] * num_variants, dtype="S"))
        neg_h5.create_dataset("alt", data=np.array(["C"] * num_variants, dtype="S"))

    # run classify

    cmd = "westminster_classify.py %s %s" % (pos_h5f, neg_h5f)
    cmd += " -o %s --stat SAD" % out_dir
    subprocess.call(cmd, shell=True)

    # check auroc
    aurocs = np.load("%s/aurocs.npy" % out_dir)
    assert aurocs.mean() > 0.8


def write_scores(h5_file, datasets):
    """Write a tiny scores HDF5 with the given {key: array} datasets."""
    with h5py.File(h5_file, "w") as h5:
        for key, data in datasets.items():
            dtype = "int64" if key == "snp_idx" else "float16"
            h5.create_dataset(key, data=np.asarray(data), dtype=dtype)


def test_read_scores_pair_max_and_sum():
    # covgene/ pair rows collapse to one feature vector per SNP
    # snp 0 -> rows 0,1 ; snp 1 -> row 2
    covgene = np.array([[1.0, 8.0], [5.0, 2.0], [3.0, 4.0]])
    snp_idx = np.array([0, 0, 1])
    h5_file = "tests/data/covgene.h5"
    write_scores(h5_file, {"covgene/logFC": covgene, "snp_idx": snp_idx})

    scores_max = read_scores(h5_file, ["covgene/logFC"], None, gene_agg="max")
    np.testing.assert_array_equal(scores_max, [[5.0, 8.0], [3.0, 4.0]])

    scores_sum = read_scores(h5_file, ["covgene/logFC"], None, gene_agg="sum")
    np.testing.assert_array_equal(scores_sum, [[6.0, 10.0], [3.0, 4.0]])


def test_read_scores_pair_unsorted_snp_idx():
    # collapsed rows are ordered by ascending SNP index regardless of row order
    covgene = np.array([[3.0, 4.0], [1.0, 8.0], [5.0, 2.0]])
    snp_idx = np.array([1, 0, 0])
    h5_file = "tests/data/covgene_unsorted.h5"
    write_scores(h5_file, {"covgene/logFC": covgene, "snp_idx": snp_idx})

    scores = read_scores(h5_file, ["covgene/logFC"], None, gene_agg="max")
    np.testing.assert_array_equal(scores, [[5.0, 8.0], [3.0, 4.0]])


def test_read_scores_mixed_families_zero_fill():
    # cov/ (3 SNPs) + covgene/ (genes only on SNPs 0 and 2): pair key reindexed
    # onto the full SNP set, geneless SNP 1 zero-filled
    cov = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    covgene = np.array([[10.0], [40.0], [70.0]])
    snp_idx = np.array([0, 2, 2])
    h5_file = "tests/data/mixed.h5"
    write_scores(
        h5_file, {"cov/logD2": cov, "covgene/logFC": covgene, "snp_idx": snp_idx}
    )

    scores = read_scores(h5_file, ["cov/logD2", "covgene/logFC"], None, gene_agg="max")
    # cov columns unchanged; covgene column maxed per SNP, 0 where no gene rows
    expected = np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 0.0], [3.0, 3.0, 70.0]])
    np.testing.assert_array_equal(scores, expected)


def test_read_scores_abs_before_max():
    # abs_value is applied before the max so the strongest-magnitude gene wins
    covgene = np.array([[-9.0], [4.0]])
    snp_idx = np.array([0, 0])
    h5_file = "tests/data/covgene_abs.h5"
    write_scores(h5_file, {"covgene/logFC": covgene, "snp_idx": snp_idx})

    scores = read_scores(
        h5_file, ["covgene/logFC"], None, abs_value=True, gene_agg="max"
    )
    np.testing.assert_array_equal(scores, [[9.0]])


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
