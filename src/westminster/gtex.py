import glob
import os
import sys

import h5py
import numpy as np
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

    for targets_name in ("targets_covgene.txt", "targets_cov.txt"):
        targets_file = gtex_scores_file.replace("scores.h5", targets_name)
        if os.path.isfile(targets_file):
            targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)
            if len(targets_df) == covgene_depth:
                return targets_name

    raise ValueError(
        f"No targets table matches {score_key} width {covgene_depth} in "
        f"{os.path.dirname(gtex_scores_file)}"
    )


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
        tissue_label = os.path.basename(path)[: -len(suffix)]
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
