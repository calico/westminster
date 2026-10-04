import pandas as pd

from westminster.gtex import gtex_keywords, match_tissue_targets, tissue_keywords

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


def test_default_prefers_smtsd():
    def match(descriptions, tissue, coarse=False):
        kw = tissue_keywords(coarse, txrev=True)[tissue]
        return match_tissue_targets(targets(descriptions), kw).tolist()

    merged = MERGED + ["brain_hippocampus"]
    assert match(merged, "Brain_Cortex") == [5]
    assert match(merged, "Brain_Cortex", coarse=True) == [5, 6]
    # no SMTSD track: single-individual targets and txrev labels fall back
    assert match(OLD, "Brain_Cortex") == [9, 10, 11]
    assert match(OLD, "Artery_Tibial") == [6, 7, 8]
    assert match(OLD, "brain_amygdala") == [9, 10, 11]


def test_pool_annotations_loaded_once_per_shared_tissue(tmp_path, monkeypatch):
    from westminster import gtex

    dirs = [tmp_path / str(i) for i in range(3)]
    calls = []
    def attributes(vcf_dir, tissue, fields):
        calls.append(tissue)
        return pd.DataFrame({'REGION': ['TSS']}, index=pd.Index(['v1'], name='variant'))
    monkeypatch.setattr(gtex, 'load_match_attributes', attributes)
    for i, directory in enumerate(dirs):
        directory.mkdir()
        for tissue in ['A', 'B', f'only_{i}']:
            pd.DataFrame({'variant': ['v1'], 'pred': [i]}).set_index('variant').to_csv(
                directory / f'{tissue}.tsv', sep='\t')
    tissues, pools = gtex.load_qtl_pools('unused', ['REGION'], *dirs)
    assert tissues == ['A', 'B']
    assert calls == ['A', 'B']
    for i, pool in enumerate(pools):
        assert pool['A'].loc['v1', 'pred'] == i
        assert pool['B'].loc['v1', 'REGION'] == 'TSS'
