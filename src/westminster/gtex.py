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
