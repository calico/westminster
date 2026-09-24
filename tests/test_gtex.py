import pandas as pd

from westminster.gtex import gtex_keywords, match_tissue_targets

# SMTS-described targets, as in every target set before the per-SMTSD merge
OLD = ["heart"] * 3 + ["blood"] * 3 + ["blood_vessel"] * 3 + ["brain"] * 3
# per-SMTSD merged targets
MERGED = ["heart_left_ventricle", "whole_blood", "artery_aorta",
          "artery_coronary", "artery_tibial", "brain_cortex"]


def targets(descriptions):
    return pd.DataFrame({
        "identifier": [f"GTEX-{i}" for i in range(len(descriptions))],
        "description": [f"RNA:{d}" for d in descriptions],
    })


def matched(descriptions, tissue):
    return match_tissue_targets(targets(descriptions), gtex_keywords[tissue]).tolist()


def test_old_targets():
    for artery in ("Artery_Aorta", "Artery_Coronary", "Artery_Tibial"):
        assert matched(OLD, artery) == [6, 7, 8]
    assert matched(OLD, "Heart_Left_Ventricle") == [0, 1, 2]
    assert matched(OLD, "Whole_Blood") == [3, 4, 5]
    assert matched(OLD, "Brain_Cortex") == [9, 10, 11]


def test_merged_targets():
    assert matched(MERGED, "Artery_Aorta") == [2]
    assert matched(MERGED, "Artery_Coronary") == [3]
    assert matched(MERGED, "Artery_Tibial") == [4]
    assert matched(MERGED, "Heart_Left_Ventricle") == [0]
    assert matched(MERGED, "Whole_Blood") == [1]
