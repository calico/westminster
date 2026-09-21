import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import pybedtools

txrev_keywords = {
    "GTEx_txrev_LCL": "lcl",
    "GTEx_txrev_adipose_subcutaneous": "adipose",
    "GTEx_txrev_adipose_visceral": "adipose",
    "GTEx_txrev_adrenal_gland": "adrenal_gland",
    "GTEx_txrev_artery_aorta": "heart",
    "GTEx_txrev_artery_coronary": "heart",
    "GTEx_txrev_artery_tibial": "heart",
    "GTEx_txrev_blood": "blood",
    "GTEx_txrev_brain_amygdala": "brain",
    "GTEx_txrev_brain_anterior_cingulate_cortex": "brain",
    "GTEx_txrev_brain_caudate": "brain",
    "GTEx_txrev_brain_cerebellar_hemisphere": "brain",
    "GTEx_txrev_brain_cerebellum": "brain",
    "GTEx_txrev_brain_cortex": "brain",
    "GTEx_txrev_brain_frontal_cortex": "brain",
    "GTEx_txrev_brain_hippocampus": "brain",
    "GTEx_txrev_brain_hypothalamus": "brain",
    "GTEx_txrev_brain_nucleus_accumbens": "brain",
    "GTEx_txrev_brain_putamen": "brain",
    "GTEx_txrev_brain_spinal_cord": "brain",
    "GTEx_txrev_brain_substantia_nigra": "brain",
    "GTEx_txrev_breast": "breast",
    "GTEx_txrev_colon_sigmoid": "colon",
    "GTEx_txrev_colon_transverse": "colon",
    "GTEx_txrev_esophagus_gej": "esophagus",
    "GTEx_txrev_esophagus_mucosa": "esophagus",
    "GTEx_txrev_esophagus_muscularis": "esophagus",
    "GTEx_txrev_fibroblast": "fibroblast",
    "GTEx_txrev_heart_atrial_appendage": "heart",
    "GTEx_txrev_heart_left_ventricle": "heart",
    "GTEx_txrev_kidney_cortex": "kidney",
    "GTEx_txrev_liver": "liver",
    "GTEx_txrev_lung": "lung",
    "GTEx_txrev_minor_salivary_gland": "salivary",
    "GTEx_txrev_muscle": "muscle",
    "GTEx_txrev_nerve_tibial": "nerve",
    "GTEx_txrev_ovary": "ovary",
    "GTEx_txrev_pancreas": "pancreas",
    "GTEx_txrev_pituitary": "pituitary",
    "GTEx_txrev_prostate": "prostate",
    "GTEx_txrev_skin_not_sun_exposed": "skin",
    "GTEx_txrev_skin_sun_exposed": "skin",
    "GTEx_txrev_small_intestine": "small_intestine",
    "GTEx_txrev_spleen": "spleen",
    "GTEx_txrev_stomach": "stomach",
    "GTEx_txrev_testis": "testis",
    "GTEx_txrev_thyroid": "thyroid",
    "GTEx_txrev_uterus": "uterus",
    "GTEx_txrev_vagina": "vagina",
}


gtex_keywords = {
    "Adipose_Subcutaneous": "adipose",
    "Adipose_Visceral_Omentum": "adipose",
    "Adrenal_Gland": "adrenal_gland",
    "Artery_Aorta": "heart",
    "Artery_Coronary": "heart",
    "Artery_Tibial": "heart",
    "Bladder": "bladder",
    "Brain_Amygdala": "brain",
    "Brain_Anterior_cingulate_cortex_BA24": "brain",
    "Brain_Caudate_basal_ganglia": "brain",
    "Brain_Cerebellar_Hemisphere": "brain",
    "Brain_Cerebellum": "brain",
    "Brain_Cortex": "brain",
    "Brain_Frontal_Cortex_BA9": "brain",
    "Brain_Hippocampus": "brain",
    "Brain_Hypothalamus": "brain",
    "Brain_Nucleus_accumbens_basal_ganglia": "brain",
    "Brain_Putamen_basal_ganglia": "brain",
    "Brain_Spinal_cord_cervical_c-1": "brain",
    "Brain_Substantia_nigra": "brain",
    "Breast_Mammary_Tissue": "breast",
    "Cells_Cultured_fibroblasts": "fibroblast",
    "Cells_EBV-transformed_lymphocytes": "lcl",
    "Colon_Sigmoid": "colon",
    "Colon_Transverse": "colon",
    "Esophagus_Gastroesophageal_Junction": "esophagus",
    "Esophagus_Mucosa": "esophagus",
    "Esophagus_Muscularis": "esophagus",
    "Heart_Atrial_Appendage": "heart",
    "Heart_Left_Ventricle": "heart",
    "Kidney_Cortex": "kidney",
    "Liver": "liver",
    "Lung": "lung",
    "Minor_Salivary_Gland": "salivary",
    "Muscle_Skeletal": "muscle",
    "Nerve_Tibial": "nerve",
    "Ovary": "ovary",
    "Pancreas": "pancreas",
    "Pituitary": "pituitary",
    "Prostate": "prostate",
    "Skin_Not_Sun_Exposed_Suprapubic": "skin",
    "Skin_Sun_Exposed_Lower_leg": "skin",
    "Small_Intestine_Terminal_Ileum": "small_intestine",
    "Spleen": "spleen",
    "Stomach": "stomach",
    "Testis": "testis",
    "Thyroid": "thyroid",
    "Uterus": "uterus",
    "Vagina": "vagina",
    "Whole_Blood": "blood",
}


# One merged per-SMTSD track per QTL tissue, for target sets built on the
# tillage 9-1 GTEx merge (seqnn 9-3 onward). Those descriptions are RNA:<smtsd>,
# so these stay plain substrings and match_tissue_targets is unchanged.
#
# Unlike gtex_keywords these are true matches: each Brain_* region hits its own
# track rather than the 13-region average, and Artery_* hits an artery track
# rather than heart. Against a single-individual target set they match nothing,
# which is the intent -- that set cannot resolve SMTSD.
gtex_smtsd_keywords = {
    "Adipose_Subcutaneous": "adipose_subcutaneous",
    "Adipose_Visceral_Omentum": "adipose_visceral_omentum",
    "Adrenal_Gland": "adrenal_gland",
    "Artery_Aorta": "artery_aorta",
    "Artery_Coronary": "artery_coronary",
    "Artery_Tibial": "artery_tibial",
    "Bladder": "bladder",
    "Brain_Amygdala": "brain_amygdala",
    "Brain_Anterior_cingulate_cortex_BA24": "brain_anterior_cingulate_cortex_ba24",
    "Brain_Caudate_basal_ganglia": "brain_caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere": "brain_cerebellar_hemisphere",
    "Brain_Cerebellum": "brain_cerebellum",
    "Brain_Cortex": "brain_cortex",
    "Brain_Frontal_Cortex_BA9": "brain_frontal_cortex_ba9",
    "Brain_Hippocampus": "brain_hippocampus",
    "Brain_Hypothalamus": "brain_hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia": "brain_nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia": "brain_putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1": "brain_spinal_cord_cervical_c_1",
    "Brain_Substantia_nigra": "brain_substantia_nigra",
    "Breast_Mammary_Tissue": "breast_mammary_tissue",
    "Cells_Cultured_fibroblasts": "cells_cultured_fibroblasts",
    # the track is cells_lcl_ebv_transformed_lymphocytes
    "Cells_EBV-transformed_lymphocytes": "ebv_transformed_lymphocytes",
    "Colon_Sigmoid": "colon_sigmoid",
    "Colon_Transverse": "colon_transverse",
    "Esophagus_Gastroesophageal_Junction": "esophagus_gastroesophageal_junction",
    "Esophagus_Mucosa": "esophagus_mucosa",
    "Esophagus_Muscularis": "esophagus_muscularis",
    "Heart_Atrial_Appendage": "heart_atrial_appendage",
    "Heart_Left_Ventricle": "heart_left_ventricle",
    "Kidney_Cortex": "kidney_cortex",
    "Liver": "liver",
    "Lung": "lung",
    "Minor_Salivary_Gland": "minor_salivary_gland",
    "Muscle_Skeletal": "muscle_skeletal",
    "Nerve_Tibial": "nerve_tibial",
    "Ovary": "ovary",
    "Pancreas": "pancreas",
    "Pituitary": "pituitary",
    "Prostate": "prostate",
    "Skin_Not_Sun_Exposed_Suprapubic": "skin_not_sun_exposed_suprapubic",
    "Skin_Sun_Exposed_Lower_leg": "skin_sun_exposed_lower_leg",
    "Small_Intestine_Terminal_Ileum": "small_intestine_terminal_ileum",
    "Spleen": "spleen",
    "Stomach": "stomach",
    "Testis": "testis",
    "Thyroid": "thyroid",
    "Uterus": "uterus",
    "Vagina": "vagina",
    "Whole_Blood": "whole_blood",
}


def tissue_keywords(smtsd: bool = False, txrev: bool = False):
    """Return the tissue -> target keyword map for a scoring pass.

    Args:
        smtsd (bool): Use the 1:1 per-SMTSD map instead of the coarse one.
        txrev (bool): Also accept the txrev tissue labels. Only meaningful for
            the coarse map; txrev labels have no SMTSD resolution, so under
            smtsd they are left out and discover_tissues reports them skipped
            rather than silently matching a coarse pool.

    Returns:
        dict: tissue label -> keyword.
    """
    if smtsd:
        return dict(gtex_smtsd_keywords)
    keywords = {}
    if txrev:
        keywords.update(
            {t.replace("GTEx_txrev_", ""): kw for t, kw in txrev_keywords.items()}
        )
    keywords.update(gtex_keywords)
    return keywords


def match_tissue_targets(targets_df, keyword, gene_targets=False, verbose=False):
    """Return array of target indices matching a GTEx tissue keyword."""
    target_ids = targets_df.identifier.values
    target_labels = targets_df.description.values
    match_tis = []
    for ti, (tid, tlab) in enumerate(zip(target_ids, target_labels)):
        tlab = tlab.lower()
        if keyword in tlab and ("GTEX" in tid or gene_targets):
            if not (keyword == "blood" and "vessel" in tlab):
                if verbose:
                    print(ti, tid, tlab)
                match_tis.append(ti)
    return np.array(match_tis)


def covgene_targets_name(gtex_scores_file: str, score_key: str):
    """Return the targets filename indexing a covgene/ stat's track axis.

    Current scoring writes covgene/ datasets over the gene-track subset of the
    strand-collapsed targets, indexed by targets_covgene.txt (baskerville
    snps.py, `targets_out_df[gene_mask_strand]`). Runs predating that wrote
    covgene/ at full strand-collapsed width, indexed by targets_cov.txt, and
    have no targets_covgene.txt at all -- and `--metrics_only` reruns never
    re-split, so the file cannot appear retroactively. Pick whichever table
    matches the stored width so both layouts read correctly.

    TODO(deprecate): once no covgene/ scores predating the targets_covgene.txt
    split (westminster 5f643a7, 2026-05-31) are still in use, drop this probe
    and go back to naming targets_covgene.txt unconditionally.

    Args:
        gtex_scores_file (str): Path to a tissue scores.h5.
        score_key (str): Score key, e.g. covgene/logFC.

    Returns:
        str: Targets filename to read alongside scores.h5.
    """
    with h5py.File(gtex_scores_file, "r") as h5_file:
        covgene_depth = h5_file[score_key].shape[-1]

    scores_dir = os.path.dirname(gtex_scores_file)
    for targets_name in ("targets_covgene.txt", "targets_cov.txt"):
        targets_file = os.path.join(scores_dir, targets_name)
        if os.path.isfile(targets_file):
            # count rows without a full parse; the caller re-reads the winner
            with open(targets_file) as targets_open:
                num_targets = sum(1 for _ in targets_open) - 1  # minus header
            if num_targets == covgene_depth:
                return targets_name

    # not ValueError: callers catch that from add_scores to skip a tissue with
    # unmatched targets, which would silently swallow a genuine layout error
    raise RuntimeError(
        f"No targets table matches {score_key} width {covgene_depth} in "
        f"{scores_dir}"
    )


def read_targets(gtex_scores_file: str, score_key: str):
    """Read the targets table indexing a score's track axis.

    Args:
        gtex_scores_file (str): Path to a scores.h5.
        score_key (str): Score key, e.g. covgene/logFC.

    Returns:
        (pd.DataFrame, bool): Targets table, and whether these are gene targets
            (which match_tissue_targets treats differently).
    """
    if score_key.startswith("gene/"):
        targets_name, gene_targets = "targets_gene.txt", True
    elif score_key.startswith("covgene/"):
        # covgene/ stats span the gene-track subset, indexed by targets_covgene.txt
        # (older runs used the full strand-collapsed set; probe to tell them apart)
        targets_name = covgene_targets_name(gtex_scores_file, score_key)
        gene_targets = False
    else:
        targets_name, gene_targets = "targets_cov.txt", False
    targets_file = gtex_scores_file.replace("scores.h5", targets_name)
    return pd.read_csv(targets_file, sep="\t", index_col=0), gene_targets


def discover_tissues(dir_glob, suffix, keyword_lookup):
    """Discover tissues from files/dirs on disk and map them to keywords.

    Args:
        dir_glob: glob pattern matching one file/dir per tissue.
        suffix: suffix to strip from each match's basename to get the tissue label.
        keyword_lookup: dict mapping tissue label to matching keyword.

    Yields:
        (tissue_label, keyword) pairs for labels found in keyword_lookup.
    """
    for path in sorted(glob.glob(dir_glob)):
        tissue_label = os.path.basename(path).removesuffix(suffix)
        if tissue_label == "merge":
            continue
        keyword = keyword_lookup.get(tissue_label)
        if keyword is None:
            print(f"Skipping {tissue_label}: no keyword mapping.", file=sys.stderr)
            continue
        yield tissue_label, keyword


def trim_dot(gene_id):
    """Trim dot off GENCODE id's."""
    dot_i = gene_id.rfind(".")
    if dot_i != -1:
        gene_id = gene_id[:dot_i]
    return gene_id


def read_gene_tss(genes_bed_file: str):
    """Build a lookup from trimmed gene ID to TSS position.

    Args:
        genes_bed_file: BED file with gene TSS positions.
            Column 3 format: ENST.../ENSG.../SYMBOL

    Returns:
        Dictionary mapping trimmed ENSG ID to (chrom, tss_pos).
    """
    gene_tss = {}
    for line in open(genes_bed_file):
        fields = line.strip().split("\t")
        chrom = fields[0]
        tss_pos = (int(fields[1]) + int(fields[2])) // 2
        name = fields[3]
        ensg = name.split("/")[1]
        ensg_trim = trim_dot(ensg)
        if ensg_trim not in gene_tss:
            gene_tss[ensg_trim] = (chrom, tss_pos)
    return gene_tss


def variant_pos(variant_id: str):
    """Parse chromosome and position from variant ID (e.g. chr1_13550_G_A_b38)."""
    parts = variant_id.split("_")
    return parts[0], int(parts[1])


def vcf_info_dist(vcf_file: str, tag: str):
    """Return {variant_id: distance} parsed from a VCF INFO field tag.

    Args:
        vcf_file: VCF file path (supports ##header lines).
        tag: INFO tag name to extract (e.g. 'SD' or 'PD').

    Returns:
        Dictionary mapping variant ID to integer distance.
    """
    dist_map = {}
    prefix = f"{tag}="
    for line in open(vcf_file):
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        variant = fields[2]
        info = fields[7]
        for field in info.split(";"):
            if field.startswith(prefix):
                dist_map[variant] = int(field[len(prefix) :])
                break
    return dist_map


_INFO_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^;\s]+)")


def parse_vcf_info(vcf_file: str, fields):
    """Parse selected INFO fields from a VCF. Returns DataFrame indexed by variant ID.

    Numeric-looking values are coerced to float; '.' becomes NaN.
    """
    rows = []
    with open(vcf_file) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            kv = dict(_INFO_RE.findall(parts[7]))
            row = {"variant": parts[2]}
            for f in fields:
                v = kv.get(f)
                if v is None or v == ".":
                    row[f] = np.nan
                else:
                    try:
                        row[f] = float(v)
                    except ValueError:
                        row[f] = v
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("variant").set_index("variant")


def load_match_attributes(vcf_dir: str, tissue: str, fields):
    """Per-variant DataFrame of positive-only INFO fields, broadcast from each
    positive to its matched negative via {tissue}_matches.tsv.

    Every attribute is a property of the *positive*; a matched negative inherits
    its positive's values, so the pair always lands in the same stratum and a
    classification metric stays paired inside it.

    Returns a DataFrame indexed by variant; suitable for df.join(...).

    A tissue the benchmark does not cover returns an empty frame, so it joins as
    NaN and drops out of any stratification. A `vcf_dir` that is not a directory
    raises instead: every tissue would take that path, and the caller would get a
    silently unstratifiable pool rather than a bad path.
    """
    if not os.path.isdir(vcf_dir):
        raise FileNotFoundError(f"QTL benchmark directory not found: {vcf_dir}")
    pos_vcf = f"{vcf_dir}/{tissue}_pos.vcf"
    matches_path = f"{vcf_dir}/{tissue}_matches.tsv"
    if not os.path.isfile(pos_vcf) or not os.path.isfile(matches_path):
        return pd.DataFrame(columns=fields)
    info = parse_vcf_info(pos_vcf, fields)
    matches = pd.read_csv(
        matches_path, sep="\t", usecols=["pos_variant", "neg_variant"]
    )
    pos_rows = info.reindex(matches["pos_variant"]).copy()
    pos_rows.index = matches["pos_variant"].values
    neg_rows = info.reindex(matches["pos_variant"]).copy()
    neg_rows.index = matches["neg_variant"].values
    out = pd.concat([pos_rows, neg_rows])
    out.index.name = "variant"
    return out[~out.index.duplicated(keep="first")]


def _tissues_in_dir(metric_dir: str):
    """Stems of *.tsv files in metric_dir, excluding files named 'metrics*'."""
    return {
        os.path.basename(f)[:-4]
        for f in glob.glob(f"{metric_dir}/*.tsv")
        if not os.path.basename(f).startswith("metrics")
    }


def load_qtl_pools(vcf_dir: str, fields, *metric_dirs):
    """Load per-tissue annotated QTL tables from one or more metric directories.

    Each metric directory is one model's `westminster_{eqtl,sqtl,paqtl}_gtex`
    output: a `{tissue}.tsv` per tissue carrying variant, label, coef, pred.

    Returns (sorted shared tissue list, [pool dict per directory]) where each
    pool maps tissue -> DataFrame with the positive-only INFO `fields` joined
    via `load_match_attributes` (and broadcast to matched negatives).
    """
    tissues = sorted(set.intersection(*[_tissues_in_dir(d) for d in metric_dirs]))
    pools = []
    for d in metric_dirs:
        pool = {}
        for t in tissues:
            df = pd.read_csv(f"{d}/{t}.tsv", sep="\t", index_col=0)
            pool[t] = df.join(load_match_attributes(vcf_dir, t, fields), on="variant")
        pools.append(pool)
    return tissues, pools


def vcf_tss_dist(vcf_file, genes_bed_file):
    """Return distance to nearest TSS for each variant, preserving VCF order."""
    genes_bt = pybedtools.BedTool(genes_bed_file)
    vcf_bt = pybedtools.BedTool(vcf_file)

    snp_order = [vc[2] for vc in vcf_bt]
    # closest requires sorted input; t='first' avoids duplicate rows on ties
    dist_hash = {
        vc[2]: (np.nan if int(vc[-1]) == -1 else int(vc[-1]))
        for vc in vcf_bt.sort(header=True).closest(genes_bt, d=True, t="first")
    }
    distances = np.array([dist_hash[v] for v in snp_order])
    assert distances.shape[0] == len(snp_order)
    return distances
