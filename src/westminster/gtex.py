import numpy as np
import pybedtools

tissue_keywords = {
    "Adipose_Subcutaneous": "adipose",
    "Adipose_Visceral_Omentum": "adipose",
    "Adrenal_Gland": "adrenal_gland",
    "Artery_Aorta": "heart",
    "Artery_Tibial": "heart",
    "Brain_Cerebellum": "brain",
    "Brain_Cortex": "brain",
    "Breast_Mammary_Tissue": "breast",
    "Colon_Sigmoid": "colon",
    "Colon_Transverse": "colon",
    "Esophagus_Mucosa": "esophagus",
    "Esophagus_Muscularis": "esophagus",
    "Liver": "liver",
    "Lung": "lung",
    "Muscle_Skeletal": "muscle",
    "Nerve_Tibial": "nerve",
    "Ovary": "ovary",
    "Pancreas": "pancreas",
    "Pituitary": "pituitary",
    "Prostate": "prostate",
    "Skin_Not_Sun_Exposed_Suprapubic": "skin",
    "Spleen": "spleen",
    "Stomach": "stomach",
    "Testis": "testis",
    "Thyroid": "thyroid",
    "Whole_Blood": "blood",
}

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


def trim_dot(gene_id):
    """Trim dot off GENCODE id's."""
    dot_i = gene_id.rfind(".")
    if dot_i != -1:
        gene_id = gene_id[:dot_i]
    return gene_id


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
