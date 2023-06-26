#!/usr/bin/env python
import argparse
import os
import pdb
import re

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

'''
westminster_gtexg_coef.py

Evaluate concordance of variant effect prediction sign classifcation
and coefficient correlations.
'''

################################################################################
# main
################################################################################
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out_dir',
                      default='coef_out',
                      help='Output directory for tissue metrics')
  parser.add_argument('-g', '--gtex_vcf_dir',
                      default='/home/drk/seqnn/data/gtex_fine/susie_pip90',
                      help='GTEx VCF directory')
  parser.add_argument('-p', '--plot',
                      action='store_true',
                      help='Generate tissue prediction plots')
  parser.add_argument('-s', '--snp_stat',
                      default='logSED',
                      help='SNP statistic. [Default: %(default)s]')
  parser.add_argument('-v', '--verbose',
                      action='store_true')
  parser.add_argument('gtex_dir')                      
  args = parser.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  
  tissue_keywords = {
      'Adipose_Subcutaneous': 'adipose',
      'Adipose_Visceral_Omentum': 'adipose',
      'Adrenal_Gland': 'adrenal_gland',
      'Artery_Aorta': 'heart',
      'Artery_Tibial': 'heart',
      'Brain_Cerebellum': 'brain',
      'Brain_Cortex': 'brain',
      'Breast_Mammary_Tissue': 'breast',
      'Colon_Sigmoid': 'colon',
      'Colon_Transverse': 'colon',
      'Esophagus_Mucosa': 'esophagus',
      'Esophagus_Muscularis': 'esophagus',
      'Liver': 'liver',
      'Lung': 'lung',
      'Muscle_Skeletal': 'muscle',
      'Nerve_Tibial': 'nerve',
      'Ovary': 'ovary',
      'Pancreas': 'pancreas',
      'Pituitary': 'pituitary',
      'Prostate': 'prostate',
      'Skin_Not_Sun_Exposed_Suprapubic': 'skin',
      'Spleen': 'spleen',
      'Stomach': 'stomach',
      'Testis': 'testis',
      'Thyroid': 'thyroid',
      'Whole_Blood': 'blood'
  }
  # 'Cells_Cultured_fibroblasts': 'fibroblast',
 
  metrics_tissue = []
  metrics_sauroc = []
  metrics_cauroc = []
  metrics_r = []
  for tissue, keyword in tissue_keywords.items():
    if args.verbose: print(tissue)

    # read causal variants
    eqtl_df = read_eqtl(tissue, args.gtex_vcf_dir)
    if eqtl_df is not None:
      # read model predictions
      gtex_scores_file = f'{args.gtex_dir}/{tissue}_pos/sed.h5'
      eqtl_df = add_scores(gtex_scores_file, keyword, eqtl_df,
                           args.snp_stat, verbose=args.verbose)

      # compute AUROCs
      sign_auroc = roc_auc_score(eqtl_df.coef > 0, eqtl_df.score)

      # compute SpearmanR
      coef_r = spearmanr(eqtl_df.coef, eqtl_df.score)[0]

      # classification AUROC
      class_auroc = classify_auroc(gtex_scores_file, keyword, eqtl_df,
                                   args.snp_stat)

      if args.plot:
        eqtl_df.to_csv(f'{args.out_dir}/{tissue}.tsv',
                       index=False, sep='\t')

        # scatterplot
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=eqtl_df.coef, y=eqtl_df.score,
                        alpha=0.5, s=20)
        plt.gca().set_xlabel('eQTL coefficient')
        plt.gca().set_ylabel('Variant effect prediction')
        plt.savefig(f'{args.out_dir}/{tissue}.png', dpi=300)

      # save
      metrics_tissue.append(tissue)
      metrics_sauroc.append(sign_auroc)
      metrics_cauroc.append(class_auroc)
      metrics_r.append(coef_r)

      if args.verbose: print('')

  # save metrics
  metrics_df = pd.DataFrame({
      'tissue': metrics_tissue,
      'auroc_sign': metrics_sauroc,
      'spearmanr': metrics_r,
      'auroc_class': metrics_cauroc
  })
  metrics_df.to_csv(f'{args.out_dir}/metrics.tsv',
                    sep='\t', index=False, float_format='%.4f')

  # summarize
  print('Sign AUROC:  %.4f' % np.mean(metrics_df.auroc_sign))
  print('SpearmanR:   %.4f' % np.mean(metrics_df.spearmanr))
  print('Class AUROC: %.4f' % np.mean(metrics_df.auroc_class))


def read_eqtl(tissue: str, gtex_vcf_dir: str, pip_t: float=0.9):
  """Reads eQTLs from SUSIE output.
  
  Args:
    tissue (str): Tissue name.
    gtex_vcf_dir (str): GTEx VCF directory.
    pip_t (float): PIP threshold.

  Returns:
    eqtl_df (pd.DataFrame): eQTL dataframe, or None if tissue skipped.
  """
  susie_dir = '/home/drk/seqnn/data/gtex_fine/tissues_susie'

  # read causal variants
  eqtl_file = f'{susie_dir}/{tissue}.tsv'
  df_eqtl = pd.read_csv(eqtl_file, sep='\t', index_col=0)

  # pip filter
  pip_match = re.search(r"_pip(\d+).?$", gtex_vcf_dir).group(1)
  pip_t = float(pip_match) / 100
  assert(pip_t > 0 and pip_t <= 1)
  df_causal = df_eqtl[df_eqtl.pip > pip_t]
  
  # make table
  tissue_vcf_file = f'{gtex_vcf_dir}/{tissue}_pos.vcf'
  if not os.path.isfile(tissue_vcf_file):
    eqtl_df = None
  else:
    # create dataframe
    eqtl_df = pd.DataFrame({
      'variant': df_causal.variant,
      'gene': [trim_dot(gene_id) for gene_id in df_causal.gene],
      'coef': df_causal.beta_marginal,
      'allele1': df_causal.allele1
    })
  return eqtl_df


def add_scores(gtex_scores_file: str,
               keyword: str,
               eqtl_df: pd.DataFrame,
               score_key: str='SED',
               verbose: bool=False):
  """Read eQTL RNA predictions for the given tissue.
  
  Args:
    gtex_scores_file (str): Variant scores HDF5.
    tissue_keyword (str): tissue keyword, for matching GTEx targets
    eqtl_df (pd.DataFrame): eQTL dataframe
    score_key (str): score key in HDF5 file
    verbose (bool): Print matching targets.

  Returns:
    eqtl_df (pd.DataFrame): eQTL dataframe, with added scores
  """
  with h5py.File(gtex_scores_file, 'r') as gtex_scores_h5:
    # read data
    snp_i = gtex_scores_h5['si'][:]
    snps = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['snp']])
    ref_allele = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['ref_allele']])
    genes = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['gene']])
    target_ids = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_ids']])
    target_labels = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_labels']])
    
    # determine matching GTEx targets
    match_tis = []
    for ti in range(len(target_ids)):
      if target_ids[ti].find('GTEX') != -1 and target_labels[ti].find(keyword) != -1:
        if not keyword == 'blood' or target_labels[ti].find('vessel') == -1:
          if verbose:
            print(ti, target_ids[ti], target_labels[ti])
          match_tis.append(ti)
    match_tis = np.array(match_tis)
    
    # read scores and take mean across targets
    score_table = gtex_scores_h5[score_key][...,match_tis].mean(axis=-1, dtype='float32')
    score_table = np.arcsinh(score_table)

  # hash scores to (snp,gene)
  snpgene_scores = {}
  for sgi in range(score_table.shape[0]):
    snp = snps[snp_i[sgi]]
    gene = trim_dot(genes[sgi])
    snpgene_scores[(snp,gene)] = score_table[sgi]

  # add scores to eQTL table
  #  flipping when allele1 doesn't match
  snp_ref = dict(zip(snps, ref_allele))
  eqtl_df_scores = []
  for sgi, eqtl in eqtl_df.iterrows():
    sgs = snpgene_scores.get((eqtl.variant,eqtl.gene), 0)
    if sgs != 0 and snp_ref[eqtl.variant] != eqtl.allele1:
      sgs *= -1
    eqtl_df_scores.append(sgs)
  eqtl_df['score'] = eqtl_df_scores

  return eqtl_df


def classify_auroc(gtex_scores_file: str,
                   keyword: str,
                   eqtl_df: pd.DataFrame,
                   score_key: str='SED'):               
  """Read eQTL RNA predictions for negatives from the given tissue.
  
  Args:
    gtex_scores_file (str): Variant scores HDF5.
    tissue_keyword (str): tissue keyword, for matching GTEx targets
    eqtl_df (pd.DataFrame): eQTL dataframe
    score_key (str): score key in HDF5 file
    verbose (bool): Print matching targets.

  Returns:
    class_auroc (float): Classification AUROC.
  """
  gtex_nscores_file = gtex_scores_file.replace('_pos','_neg')
  with h5py.File(gtex_nscores_file, 'r') as gtex_scores_h5:
    # read data
    snp_i = gtex_scores_h5['si'][:]
    snps = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['snp']])
    genes = np.array([snp.decode('UTF-8') for snp in gtex_scores_h5['gene']])
    target_ids = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_ids']])
    target_labels = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_labels']])
    
    # determine matching GTEx targets
    match_tis = []
    for ti in range(len(target_ids)):
      if target_ids[ti].find('GTEX') != -1 and target_labels[ti].find(keyword) != -1:
        if not keyword == 'blood' or target_labels[ti].find('vessel') == -1:
          match_tis.append(ti)
    match_tis = np.array(match_tis)
    
    # read scores and take mean across targets
    score_table = gtex_scores_h5[score_key][...,match_tis].mean(axis=-1, dtype='float32')
    # score_table = np.arcsinh(score_table)

  # aggregate across genes w/ sum abs
  nsnp_scores = {}
  for sgi in range(score_table.shape[0]):
    snp = snps[snp_i[sgi]]
    nsnp_scores[snp] = nsnp_scores.get(snp,0) + np.abs(score_table[sgi])

  psnp_scores = {}
  for sgi, eqtl in eqtl_df.iterrows():
    snp = eqtl.variant
    psnp_scores[snp] = psnp_scores.get(snp,0) + np.abs(eqtl.score)

  # compute AUROC
  Xp = list(psnp_scores.values())
  Xn = list(nsnp_scores.values())
  X = Xp + Xn
  y = [1]*len(Xp) + [0]*len(Xn)
  
  return roc_auc_score(y, X)


def trim_dot(gene_id):
  """Trim dot off GENCODE id's."""
  dot_i = gene_id.rfind('.')
  if dot_i != -1:
    gene_id = gene_id[:dot_i]
  return gene_id


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
