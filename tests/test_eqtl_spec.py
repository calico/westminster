import sys

import pandas as pd
import pytest

from westminster.scripts import westminster_eqtl_spec as spec


class InputsReached(Exception):
    """Stop after CLI validation and tissue resolution, before loading data."""


@pytest.mark.parametrize(
    "tissues, extra_args, expected",
    [
        (list(spec.gtex_keywords), ["--drop_groups", "liver"],
         set(spec.tissue_map(True).values()) - {"liver"}),
        (["Liver", "Lung", "Brain_Cortex"], [],
         {"liver", "lung", "brain_cortex"}),
        (["Liver", "Lung", "Brain_Cortex"], ["--drop_groups", "liver"],
         {"lung", "brain_cortex"}),
    ],
)
def test_native_axis(tmp_path, monkeypatch, tissues, extra_args, expected):
    merge = tmp_path / "merge"
    merge.mkdir()
    (merge / "scores.h5").touch()
    out = tmp_path / "out"
    targets = pd.DataFrame({
        "identifier": [f"GTEX_{i}" for i in range(len(tissues))],
        "description": tissues,
    })
    monkeypatch.setattr(spec, "read_targets", lambda *args: (targets, False))

    def read_positives(directory, groups, tmap):
        assert set(groups) == expected
        assert tmap == spec.tissue_map(True)
        raise InputsReached

    monkeypatch.setattr(spec, "read_positives", read_positives)
    monkeypatch.setattr(sys, "argv", [
        "westminster_eqtl_spec", str(tmp_path), "--native", "-o", str(out),
        *extra_args,
    ])
    with pytest.raises(InputsReached):
        spec.main()
    assert set(pd.read_csv(out / "groups.tsv", sep="\t").group) == expected


def test_invalid_drop_group(tmp_path, monkeypatch, capsys):
    merge = tmp_path / "merge"
    merge.mkdir()
    (merge / "scores.h5").touch()
    monkeypatch.setattr(sys, "argv", [
        "westminster_eqtl_spec", str(tmp_path), "--native",
        "--drop_groups", "livre", "-o", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as error:
        spec.main()
    assert error.value.code == 2
    assert "--drop_groups: not tissue groups: ['livre']" in capsys.readouterr().err


def test_too_few_matched_tissues(monkeypatch):
    targets = pd.DataFrame({"identifier": ["GTEX_0"], "description": ["Liver"]})
    monkeypatch.setattr(spec, "read_targets", lambda *args: (targets, False))
    with pytest.raises(ValueError, match="Fewer than 2 resolvable tissue groups"):
        spec.resolve_groups("unused", "covgene/logFC", spec.tissue_map(True))
