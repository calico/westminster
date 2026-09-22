import errno
import os
from pathlib import Path

import pytest
import h5py
import numpy as np

from westminster.multi import gcp_mirror_dir, link_merge_scores, relocate_gcp_scores


def stage_merge(models_dir, out_dir, folds):
    pytest.importorskip("baskerville_torch.scripts.hound_snp_folds")
    for i, sub in enumerate([*folds, "ensemble"]):
        merge = models_dir / sub / out_dir / "merge"
        merge.mkdir(parents=True)
        (merge / "targets_cov.txt").write_text("targets")
        with h5py.File(merge / "scores.h5", "w") as scores:
            scores["snp"] = np.array([b"rs1"])
            scores["cov/logSUM"] = np.array([[2 * i]], dtype="float16")


def test_link_merge_repeat_preserves_links(tmp_path):
    stage_merge(tmp_path, "source", ["f0c0"])
    link_merge_scores(tmp_path, "subset", "source", ["f0c0"])
    link = tmp_path / "f0c0/subset/merge"
    inode = link.lstat().st_ino

    link_merge_scores(tmp_path, "subset", "source", ["f0c0"])

    assert link.lstat().st_ino == inode
    assert not os.path.isabs(os.readlink(link))
    with h5py.File(link / "scores.h5") as scores:
        assert scores["cov/logSUM"][0, 0] == 0


def test_link_merge_rejects_changed_source_before_any_links(tmp_path):
    for source in ["source_a", "source_b"]:
        stage_merge(tmp_path, source, ["f0c0"])
    link_merge_scores(tmp_path, "subset", "source_a", ["f0c0"])
    # A missing fold link must not be created before checking the ensemble.
    fold_link = tmp_path / "f0c0/subset/merge"
    fold_link.unlink()
    ensemble_link = tmp_path / "ensemble/subset/merge"
    metrics = ensemble_link.parent / "metrics.tsv"
    metrics.write_text("source_a metrics")

    with pytest.raises(FileExistsError, match="new output directory"):
        link_merge_scores(tmp_path, "subset", "source_b", ["f0c0"])

    assert not fold_link.exists()
    with h5py.File(ensemble_link / "scores.h5") as scores:
        assert scores["cov/logSUM"][0, 0] == 0
    assert metrics.read_text() == "source_a metrics"


@pytest.mark.parametrize("folds, expected", [(["f0c0"], 0), (["f0c0", "f1c0"], 1)])
def test_link_merge_ensembles_only_requested_folds(tmp_path, folds, expected):
    stage_merge(tmp_path, "source", ["f0c0", "f1c0", "f0c1"])
    source_ensemble = tmp_path / "ensemble/source/merge/scores.h5"
    original = source_ensemble.read_bytes()

    link_merge_scores(tmp_path, "subset", "source", folds)

    ensemble = tmp_path / "ensemble/subset/merge"
    assert not ensemble.is_symlink()
    with h5py.File(ensemble / "scores.h5") as scores:
        assert scores["cov/logSUM"][0, 0] == expected
        assert scores["snp"][0] == b"rs1"
    assert (ensemble / "targets_cov.txt").read_text() == "targets"
    assert source_ensemble.read_bytes() == original


def test_link_merge_rejects_changed_folds(tmp_path):
    stage_merge(tmp_path, "source", ["f0c0", "f1c0"])
    link_merge_scores(tmp_path, "subset", "source", ["f0c0"])

    with pytest.raises(FileExistsError, match="new output directory"):
        link_merge_scores(tmp_path, "subset", "source", ["f0c0", "f1c0"])

    assert not (tmp_path / "f1c0/subset").exists()


def test_link_merge_rejects_unverified_existing_ensemble(tmp_path):
    stage_merge(tmp_path, "source", ["f0c0"])
    ensemble = tmp_path / "ensemble/subset/merge"
    ensemble.parent.mkdir(parents=True)
    ensemble.symlink_to("../source/merge")

    with pytest.raises(FileExistsError, match="new output directory"):
        link_merge_scores(tmp_path, "subset", "source", ["f0c0"])

    assert not (tmp_path / "f0c0/subset").exists()
    assert ensemble.is_symlink()


def stage_mirror(models_dir: str, out_dir: str, fold_crosses, tag: str):
    """Write a fake GCP fetch: one scores.h5 per fold plus the ensemble."""
    mirror = gcp_mirror_dir(models_dir, out_dir)
    for sub in list(fold_crosses) + ["ensemble"]:
        os.makedirs(os.path.join(mirror, sub))
        with open(os.path.join(mirror, sub, "scores.h5"), "w") as f:
            f.write(f"{tag}/{sub}")


def test_relocate_keeps_configs_apart(tmp_path, monkeypatch):
    """Two configs fetching the same out_dir must not touch each other's scores."""
    monkeypatch.chdir(tmp_path)
    out_dir = "eqtl/merge"
    fold_crosses = ["f0c0", "f1c0"]

    models1 = str(tmp_path / "v1")
    models2 = str(tmp_path / "v2")
    stage_mirror(models1, out_dir, fold_crosses, "v1")
    stage_mirror(models2, out_dir, fold_crosses, "v2")

    relocate_gcp_scores(out_dir, models1, fold_crosses)

    # scores land in the embed layout split_scores reads back
    for sub in fold_crosses + ["ensemble"]:
        landed = os.path.join(models1, sub, out_dir, "scores.h5")
        assert open(landed).read() == f"v1/{sub}"

    # the emptied mirror is gone, and the untouched config's is intact
    assert not os.path.exists(os.path.join(models1, ".gcp_fetch"))
    assert (
        open(os.path.join(gcp_mirror_dir(models2, out_dir), "f0c0", "scores.h5")).read()
        == "v2/f0c0"
    )

    # nothing was staged in the working directory
    assert sorted(os.listdir(tmp_path)) == ["v1", "v2"]


def test_relocate_leaves_concurrent_stage_standing(tmp_path):
    """A second stage mid-fetch for the same config keeps its mirror."""
    models_dir = str(tmp_path / "v1")
    fold_crosses = ["f0c0"]
    stage_mirror(models_dir, "eqtl/merge", fold_crosses, "eqtl")
    stage_mirror(models_dir, "sqtl/merge", fold_crosses, "sqtl")

    relocate_gcp_scores("eqtl/merge", models_dir, fold_crosses)

    assert not os.path.exists(gcp_mirror_dir(models_dir, "eqtl/merge"))
    assert (
        open(
            os.path.join(gcp_mirror_dir(models_dir, "sqtl/merge"), "f0c0", "scores.h5")
        ).read()
        == "sqtl/f0c0"
    )


def test_relocate_preserves_empty_sibling_and_unexpected_files(tmp_path):
    models_dir = str(tmp_path / "v1")
    stage_mirror(models_dir, "eqtl/merge", ["f0c0"], "eqtl")
    sibling = Path(gcp_mirror_dir(models_dir, "sqtl/merge")) / "f0c0"
    sibling.mkdir(parents=True)
    extra = Path(gcp_mirror_dir(models_dir, "eqtl/merge")) / "unexpected.txt"
    extra.write_text("keep")

    relocate_gcp_scores("eqtl/merge", models_dir, ["f0c0"])

    assert sibling.is_dir()
    assert extra.read_text() == "keep"


@pytest.mark.parametrize("relative_dir", ["eqtl/merge/f0c0", "eqtl/merge", "eqtl", "."])
def test_relocate_preserves_files_arriving_during_cleanup(
    tmp_path, monkeypatch, relative_dir
):
    models_dir = str(tmp_path / "v1")
    stage_mirror(models_dir, "eqtl/merge", ["f0c0"], "eqtl")
    target = Path(models_dir) / ".gcp_fetch" / relative_dir
    arriving = target / "arriving.h5"
    original_rmdir = os.rmdir

    def concurrent_rmdir(path):
        if Path(path) == target:
            arriving.write_text("keep")
        original_rmdir(path)

    monkeypatch.setattr(os, "rmdir", concurrent_rmdir)
    relocate_gcp_scores("eqtl/merge", models_dir, ["f0c0"])

    assert arriving.read_text() == "keep"
    for sub in ["f0c0", "ensemble"]:
        assert (Path(models_dir) / sub / "eqtl/merge/scores.h5").read_text() == (
            f"eqtl/{sub}"
        )


@pytest.mark.parametrize("error", [errno.ENOENT, errno.EACCES])
def test_relocate_cleanup_errors(tmp_path, monkeypatch, error):
    models_dir = str(tmp_path / "v1")
    stage_mirror(models_dir, "eqtl/merge", ["f0c0"], "eqtl")
    target = Path(gcp_mirror_dir(models_dir, "eqtl/merge")) / "f0c0"
    original_rmdir = os.rmdir

    def concurrent_rmdir(path):
        if Path(path) == target:
            if error == errno.ENOENT:
                original_rmdir(path)
            raise OSError(error, os.strerror(error), str(path))
        original_rmdir(path)

    monkeypatch.setattr(os, "rmdir", concurrent_rmdir)
    if error == errno.ENOENT:
        relocate_gcp_scores("eqtl/merge", models_dir, ["f0c0"])
        assert not (Path(models_dir) / ".gcp_fetch").exists()
    else:
        with pytest.raises(PermissionError):
            relocate_gcp_scores("eqtl/merge", models_dir, ["f0c0"])
