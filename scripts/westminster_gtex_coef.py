#!/usr/bin/env python
import argparse
import os
import pdb

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

'''
westminster_gtex_coef.py

Evaluate concordance of variant effect prediction sign classifcation
and coefficient correlations.
'''

################################################################################
# main
################################################################################
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out_dir',
                      default='coef_out')
  parser.add_argument('-g', '--gtex_vcf_dir',
                      default='/home/drk/seqnn/data/gtex_fine/susie_pip90')
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
 
  sign_auroc_list = []
  coef_r_list = []
  for tissue, keyword in tissue_keywords.items():
    if args.verbose: print(tissue)

    # read causal variants
    eqtl_df = read_eqtl(tissue, args.gtex_vcf_dir)

    # read model predictions
    variant_scores = read_scores(tissue, keyword, args.gtex_dir,
                                 eqtl_df, verbose=args.verbose)
    variant_scores = variant_scores[eqtl_df.consistent]

    # compute AUROCs
    variant_sign = eqtl_df[eqtl_df.consistent].sign
    sign_auroc = roc_auc_score(variant_sign, variant_scores)

    # compute PearsonR
    variant_coef = eqtl_df[eqtl_df.consistent].coef
    coef_r = spearmanr(variant_coef, variant_scores)[0]

    # save
    sign_auroc_list.append(sign_auroc)
    coef_r_list.append(coef_r)

    if args.verbose: print('')

  # save metrics
  metrics_df = pd.DataFrame({
      'tissue': tissue_keywords.keys(),
      'auroc': sign_auroc_list,
      'spearmanr': coef_r_list
  })
  metrics_df.to_csv(f'{args.out_dir}/metrics.tsv',
                    sep='\t', index=False, float_format='%.4f')

  # summarize
  print('AUROC: %.4f' % np.mean(metrics_df.auroc))
  print('SpearmanR: %.4f' % np.mean(metrics_df.spearmanr))


def read_eqtl(tissue: str, gtex_vcf_dir: str, pip_t: float=0.9):
  """Reads eQTLs from SUSIE output.
  
  Args:
    tissue (str): Tissue name.
    gtex_vcf_dir (str): GTEx VCF directory.
    pip_t (float): PIP threshold.

  Returns:
    eqtl_df (pd.DataFrame): eQTL dataframe.
  """
  susie_dir = '/home/drk/seqnn/data/gtex_fine/tissues_susie'

  # read causal variants
  eqtl_file = f'{susie_dir}/{tissue}.tsv'
  df_eqtl = pd.read_csv(eqtl_file, sep='\t', index_col=0)
  df_causal = df_eqtl[df_eqtl.pip > pip_t]

  # remove variants with inconsistent signs
  variant_a1 = {}
  variant_sign = {}
  variant_z = {}
  inconsistent_variants = set()
  for variant in df_causal.itertuples():
    vid = variant.variant
    vsign = variant.z > 0

    variant_a1[vid] = variant.allele1
    variant_z.setdefault(vid,[]).append(variant.z)
    if vid in variant_sign:
      if variant_sign[vid] != vsign:
        inconsistent_variants.add(vid)
    else:
      variant_sign[vid] = vsign

  # delete inconsistent variants  
  # for vid in inconsistent_variants:
  #   del variant_sign[vid]
  #   del variant_z[vid]
  #   del variant_a1[vid]
      
  # average z-scores across genes
  for vid in variant_z:
    variant_z[vid] = np.mean(variant_z[vid])

  # order variants
  tissue_vcf_file = f'{gtex_vcf_dir}/{tissue}_pos.vcf'
  pred_variants = np.array([line.split()[2] for line in open(tissue_vcf_file) if not line.startswith('##')])
  consistent_mask = np.array([vid not in inconsistent_variants for vid in pred_variants])

  # create dataframe
  # eqtl_df = pd.DataFrame({
  #   'variant': pred_variants[sign_mask],
  #   'coef': [variant_z[vid] for vid in pred_variants[sign_mask]],
  #   'sign': [variant_sign[vid] for vid in pred_variants[sign_mask]],
  #   'allele': [variant_a1[vid] for vid in pred_variants[sign_mask]]
  # })
  eqtl_df = pd.DataFrame({
    'variant': pred_variants,
    'coef': [variant_z[vid] for vid in pred_variants],
    'sign': [variant_sign[vid] for vid in pred_variants],
    'allele': [variant_a1[vid] for vid in pred_variants],
    'consistent': consistent_mask
  })
  return eqtl_df


def read_scores(tissue: str,
                keyword: str,
                gtex_dir: str,
                eqtl_df: pd.DataFrame,
                score_key: str='SAD',
                verbose: bool=False):
  """Read eQTL RNA predictions for the given tissue.
  
  Args:
    tissue (str): tissue name
    tissue_keyword (str): tissue keyword, for matching GTEx targets
    gtex_dir (str): directory containing eQTL predictions
    eqtl_df (pd.DataFrame): eQTL dataframe
    score_key (str): score key in HDF5 file
    verbose (bool): Print matching targets.

  Returns:
    np.array: eQTL predictions
  """
  gtex_scores_file = f'{gtex_dir}/{tissue}_pos/sad.h5'
  with h5py.File(gtex_scores_file, 'r') as gtex_scores_h5:
    score_ref = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['ref_allele']])
    
    # determine matching GTEx targets
    target_ids = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_ids']])
    target_labels = np.array([ref.decode('UTF-8') for ref in gtex_scores_h5['target_labels']])
    match_tis = []
    for ti in range(len(target_ids)):
      if target_ids[ti].find('GTEX') != -1 and target_labels[ti].find(keyword) != -1:
        if not keyword == 'blood' or target_labels[ti].find('vessel') == -1:
          if verbose:
            print(ti, target_ids[ti], target_labels[ti])
          match_tis.append(ti)
    match_tis = np.array(match_tis)
    
    # mean across targets
    variant_scores = gtex_scores_h5[score_key][...,match_tis].mean(axis=-1, dtype='float32')
    variant_scores = np.arcsinh(variant_scores)

  # flip signs
  sad_flip = (score_ref != eqtl_df.allele)
  variant_scores[sad_flip] = -variant_scores[sad_flip]

  return variant_scores


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
