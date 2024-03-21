#!/usr/bin/env python

import shutil
import subprocess

import h5py
import numpy as np

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
    cmd += " -o %s" % out_dir
    subprocess.call(cmd, shell=True)

    # check auroc
    aurocs = np.load("%s/aurocs.npy" % out_dir)
    assert aurocs.mean() > 0.8


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
