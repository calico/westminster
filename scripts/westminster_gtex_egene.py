#!/usr/bin/env python

import numpy as np
import pandas as pd
import h5py
import argparse

from westminster import parse_gtf

'''
westminster_gtex_egene.py

Benchmark a Baskerville model on the GTEx eGene prioritization task.
'''


def read_h5(h5_file, snp_stat, targets_file=None):
    with h5py.File(h5_file, 'r') as borzoi_h5:
        if targets_file is not None:
            targets_df = pd.read_csv(targets_file, sep='\t', header=0, index_col=0)
            target_indices = list(targets_df.index)
            scores_df = pd.DataFrame(borzoi_h5[snp_stat][:, target_indices])
            scores_df.columns = ['RNA' + str(i) for i in range(scores_df.shape[1])]
        else:
            scores_df = pd.DataFrame(borzoi_h5[snp_stat][:, :])
            scores_df.columns = ['RNA' + str(i) for i in range(scores_df.shape[1])]

        scores_df['SNP_ID'] = list(borzoi_h5['si'])  # Each si is a SNP ID, a single si can have multiple target genes
        scores_df['target_gene'] = borzoi_h5['gene']
        scores_df['target_gene'] = scores_df['target_gene'].str.decode('utf-8')
        scores_df['mean_score'] = scores_df.filter(regex='RNA').mean(axis=1)

        for col in scores_df.columns:  # Remove all RNA columns as we only need the mean score
            if 'RNA' in col:
                del scores_df[col]

        # get variant id maps
        variant_id_map = pd.DataFrame(borzoi_h5['snp'])
        variant_id_map.reset_index(
            inplace=True)  # Get index (borzoi['si'] for these variants, equivalent to a reset_index op)
        variant_id_map.columns = ['SNP_ID', 'variant']
        variant_id_map['variant'] = variant_id_map['variant'].str.decode('utf-8')

    return scores_df, variant_id_map


def get_variant_scores(h5_file, snp_stat, genes_dict, targets_file=None):
    bz_scores_df, variant_id_map = read_h5(h5_file=h5_file,
                                           snp_stat=snp_stat,
                                           targets_file=targets_file)

    df = variant_id_map.merge(bz_scores_df, on='SNP_ID')  # Each row now has SNP, SNP_ID, Gene and Bz (RNA) scores.
    bz_gene_type = [genes_dict[x.split(".")[0]]['gene_type'] for x in df['target_gene']]
    df = df[np.array(bz_gene_type) == 'protein_coding']  # Only keep scores for protein coding genes
    return df[['SNP_ID', 'variant', 'target_gene', 'mean_score']]


def read_gtf(gtf_file):
    genes_dict = parse_gtf.parse_gtf_file(gtf_file)
    return genes_dict


def read_eqtls(f_name, genes_dict):

    finemapped_dat = pd.read_csv(f_name, sep='\t', header=0)

    print("Number of cred. sets:", finemapped_dat['molecular_trait_id'].nunique())
    # If gene associated with cred. set is not in the genes_dict, then it is missing from the GENCODE version \
    # used to generate the finemapped files. Removing cred. sets associated with missing genes.
    egene_type = []
    missing_egenes = []
    for gene in finemapped_dat['molecular_trait_id']:
        try:
            egene_type.append(genes_dict[gene]['gene_type'])
        except KeyError:
            print(egene)
            egene_type.append('missing')
            missing_egenes.append(gene)
    print("Number of cred. sets with missing genes:", len(set(missing_egenes)))

    finemapped_prot_coding = finemapped_dat[np.array(egene_type) == 'protein_coding']
    print("Number of cred. sets linked to protein-coding gene:", finemapped_prot_coding['molecular_trait_id'].nunique())
    finemapped_prot_coding = finemapped_prot_coding[['chromosome', 'position', 'variant', 'cs_id', 'pip']]
    finemapped_prot_coding = remove_indels(finemapped_prot_coding, threshold=0.1)
    return finemapped_prot_coding


def remove_indels(finemapped_dat, threshold=0.1):
    """
    Remove INDELs.
    If INDEL PIP <= threshold:
        Throw out the INDEL.
    Else:
        Throw out the INDEL + the entire cred. set.
    """
    ref_allele = [x.split('_')[-2] for x in finemapped_dat['variant']]
    alt_allele = [x.split('_')[-1] for x in finemapped_dat['variant']]
    var_max_length = np.array([max(len(x), len(y)) for x, y in zip(ref_allele, alt_allele)])
    impacted_cred_sets = finemapped_dat['cs_id'][(var_max_length > 1) & (finemapped_dat['pip'] > threshold)]
    remove_cred_set = np.array([x in impacted_cred_sets.to_list() for x in finemapped_dat['cs_id']])
    remove_var = np.array([x > 1 for x in var_max_length])
    remove_rows = np.logical_or(remove_cred_set, remove_var)
    finemapped_dat_filt = finemapped_dat[np.logical_not(remove_rows)]  # remove INDELs
    print("Number of cred. sets removed due to INDELs:", len(set(finemapped_dat['cs_id'])) -
          len(set(finemapped_dat_filt['cs_id'])))
    return finemapped_dat_filt

def assign_bz_scores(borzoi_df, fine_mapped_df):
    merged_df = fine_mapped_df.merge(borzoi_df, on=['variant'], how='inner')
    return merged_df


class Eval:

    def __init__(self):
        pass

    @staticmethod
    def top_k_eval(egene, gene_vec, pred_vec, k):
        # Assuming that predicted eGene is the gene with the max. bz score / max. dist. inverse score
        if k >= 1:
            sort_order = np.argsort(pred_vec)[::-1]
            pred_genes = np.array(gene_vec)[sort_order][:k]
            # pred_genes = [x.split('.')[0] for x in pred_genes]

            if egene in pred_genes:
                return 1
            else:
                return 0
        else:
            raise ValueError("k must be >= 1")

    @staticmethod
    def create_gene_tss_map(genes_dict):
        for gene in genes_dict.keys():
            genes_dict[gene]['tss'] = []
            for transcript in genes_dict[gene]['transcripts']:
                if transcript['transcript_type'] == 'protein_coding':
                    genes_dict[gene]['tss'].append(transcript['start'])  # strand has already accounted for in start
                else:
                    pass
        return genes_dict

    def cs_gene_scores(self, bz_scores, genes_dict, k):
        """
        How do we deal with the fact that Borzoi does not score all genes for all variants?
            Since the 196kb window around each variant will consist of different gene sets, some variant-gene \
            pairs will not be scored by Borzoi. In these cases, I assign a score of 0 to the variant-gene pairs.
            # For example:
            # variant            position  pip      chromosome  ENSG00000162631.18  ENSG00000198890.8
            # chr1_106979924_T_A 106979924 0.249720 1                          NaN           0.010590
            # chr1_107003570_T_C 107003570 0.436914 1                          NaN           0.002975
            # chr1_107086667_C_T 107086667 0.136238 1                     0.008514           0.019989
            # chr1_107135646_G_C 107135646 0.057455 1                     4.406250           0.691406

            In the above case, Borzoi does not score ENSG00000162631.18 for chr1_106979924_T_A,
            it will be assigned a score=0. This is done by using the fill_value arg. in the pd.pivot_table() call.

        For each credible set, compute distance to the subset of genes that are scored by Borzoi to increase speed.
        Raise an Exception if there are no genes in this subset.
            Note:
            # This means I am only computing distance-to-TSS for genes within a 196608kb window of \
            # at-least one variant in order to identify the nearest gene.
            # If there is any gene in this set, it will include by definition the nearest gene.
            # If there is no gene in this set, check distances for all genes to find the closest gene.
            # ^ Above is not an edge case I have never encountered, but TODO: raise an Exception incase we encounter it.
        """
        tss_dict = self.create_gene_tss_map(genes_dict=genes_dict)
        cred_sets = bz_scores['cs_id'].unique()
        pred_genes_bz = []
        pred_genes_dist = []

        for idx in range(len(cred_sets)):
            cs = bz_scores[bz_scores['cs_id'] == cred_sets[idx]] # cred_sets[idx] is the egene name + an index.
            # For example: ENSG00000156876_L1
            egene = cred_sets[idx].split("_")[0]
            dat = pd.pivot_table(cs.astype({'mean_score': np.float64}),
                                 index=['variant', 'position', 'pip', 'chromosome'],
                                 columns=['target_gene'], values='mean_score',
                                 fill_value=0)
            dat.reset_index(inplace=True)

            borzoi_scores = []
            dist_scores = []
            genes = []

            for col_name in dat.columns:
                if 'ENSG' in col_name: # If the column is a gene
                    gene_id = col_name.split('.')[0]
                    if tss_dict[gene_id]['tss']:  # If there is at least one protein-coding gene in the set
                        # compute borzoi score
                        pip_weighted_borzoi_score = sum(dat['pip'] * np.abs(dat[col_name]))
                        borzoi_scores.append(pip_weighted_borzoi_score)
                        # compute distance score for each alt. TSS
                        tss_scores = []
                        for tss in tss_dict[gene_id]['tss']:
                            dat['DIST.' + gene_id] = tss - dat['position']
                            weighted_dist_score = sum(dat['pip'] * (1 / np.abs(dat['DIST.' + gene_id])))
                            tss_scores.append(weighted_dist_score)
                        dist_scores.append(max(tss_scores))  # max. distance score across all alt. TSSs
                        genes.append(gene_id)
                    else:
                        # i.e. no protein coding transcripts exist for this gene
                        # exclude gene from analysis
                        print("No protein coding transcripts exist for gene: {}".format(col_name))

            if len(dat) == 0:
                print(egene, "has no variants for which genes were scored in credible set")
            else:
                pred_genes_bz.append(self.top_k_eval(egene, genes, borzoi_scores, k))
                pred_genes_dist.append(self.top_k_eval(egene, genes, dist_scores, k))

        topk_bz = sum(pred_genes_bz) / float(len(pred_genes_bz))
        topk_dist = sum(pred_genes_dist) / float(len(pred_genes_dist))
        return topk_bz, topk_dist


if __name__ == "__main__":

    # Required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--borzoi_sed_h5',
                        help='Hdf5 file with Borzoi SED scores')
    parser.add_argument('-e', '--eqtl',
                        help='eqtl file path')
    parser.add_argument('-g', '--gtf',
                        help='path to gencode gtf')
    parser.add_argument('-k', '--topk',
                        default=1,
                        help='top k genes to consider for evaluation')
    parser.add_argument('-s', '--snp_stat',
                        default='logD2',
                        help='SNP statistic. [Default: %(default)s]')
    parser.add_argument('-t', '--targets_file',
                        help='targets file path',
                        default=None)
    parser.add_argument('-o', '--out_file',
                        help='output file path')

    args = parser.parse_args()

    genes_dict = read_gtf(args.gtf)
    fine_mapped_eqtls = read_eqtls(args.eqtl, genes_dict=genes_dict)

    bz_df = get_variant_scores(h5_file=args.borzoi_sed_h5,
                               snp_stat=args.snp_stat,
                               genes_dict=genes_dict,
                               targets_file=args.targets_file)

    eqtl_bz_df = assign_bz_scores(borzoi_df=bz_df, fine_mapped_df=fine_mapped_eqtls)

    compute_linkage = Eval()
    acc_bz, acc_dist = compute_linkage.cs_gene_scores(bz_scores=eqtl_bz_df, genes_dict=genes_dict, k=int(args.topk))
    with open(args.out_file, 'w') as f:
        f.write("Borzoi top-{0} accuracy:{1}\n".format(args.topk, acc_bz))
        f.write("Distance top-{0} accuracy:{1}\n".format(args.topk, acc_dist))
