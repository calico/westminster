import os

from westminster.multi import gcp_mirror_dir, relocate_gcp_scores


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
