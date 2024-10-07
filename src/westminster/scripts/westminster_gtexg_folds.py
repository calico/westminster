#!/usr/bin/env python
# Copyright 2023 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser, OptionGroup
import glob
import pickle
import pdb
import os
import sys

import h5py
import numpy as np

import slurm

"""
westminster_gtexg_folds.py

Benchmark Baskerville model replicates on GTEx eQTL coefficient task.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <data_dir>"
    parser = OptionParser(usage)

    # sed options
    snp_options = OptionGroup(parser, "hound_snpgene.py options")
    snp_options.add_option(
        "-c",
        dest="cluster_pct",
        default=0,
        type="float",
        help="Cluster genes within a %% of the seq length to make a single ref pred [Default: %default]",
    )
    snp_options.add_option(
        "-f",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ["HG38"],
        help="Genome FASTA for sequences [Default: %default]",
    )
    snp_options.add_option(
        "--float16",
        dest="float16",
        default=False,
        action="store_true",
        help="Use mixed float16 precision [Default: %default]",
    )
    snp_options.add_option(
        "-g",
        dest="genes_gtf",
        default="%s/genes/gencode41/gencode41_basic_nort.gtf" % os.environ["HG38"],
        help="GTF for gene definition [Default %default]",
    )
    snp_options.add_option(
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts [Default: %default]",
    )
    snp_options.add_option(
        "-o",
        dest="out_dir",
        default="sed",
        help="Output directory for tables and plots [Default: %default]",
    )
    snp_options.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    snp_options.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    snp_options.add_option(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %default]",
    )
    snp_options.add_option(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    snp_options.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    snp_options.add_option(
        "-u", dest="untransform_old", default=False, action="store_true"
    )
    snp_options.add_option(
        "--gcs",
        dest="gcs",
        default=False,
        action="store_true",
        help="Input and output are in gcs",
    )
    snp_options.add_option(
        "--require_gpu",
        dest="require_gpu",
        default=False,
        action="store_true",
        help="Only run on GPU",
    )
    snp_options.add_option(
        "--tensorrt",
        dest="tensorrt",
        default=False,
        action="store_true",
        help="Model type is tensorrt optimized",
    )
    parser.add_option_group(snp_options)

    # cross-fold
    fold_options = OptionGroup(parser, "cross-fold options")
    fold_options.add_option(
        "--crosses",
        dest="crosses",
        default=1,
        type="int",
        help="Number of cross-fold rounds [Default:%default]",
    )
    fold_options.add_option(
        "-d",
        dest="data_head",
        default=None,
        type="int",
        help="Index for dataset/head [Default: %default]",
    )
    fold_options.add_option(
        "-e",
        dest="conda_env",
        default="tf210",
        help="Anaconda environment [Default: %default]",
    )
    fold_options.add_option(
        "--folds",
        dest="num_folds",
        default=None,
        type="int",
        help="Number of folds to evaluate [Default: %default]",
    )
    fold_options.add_option(
        "--f_list",
        dest="fold_subset_list",
        default=None,
        help="Subset of folds to evaluate (encoded as comma-separated string) [Default:%default]",
    )
    fold_options.add_option(
        "--gtex",
        dest="gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_fine/susie_pip90",
    )
    fold_options.add_option(
        "--name",
        dest="name",
        default="gtex",
        help="SLURM name prefix [Default: %default]",
    )
    fold_options.add_option(
        "--max_proc",
        dest="max_proc",
        default=None,
        type="int",
        help="Maximum concurrent processes [Default: %default]",
    )
    fold_options.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script. \
            (Unused, but needs to appear as dummy.)",
    )
    fold_options.add_option(
        "-q",
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    parser.add_option_group(fold_options)

    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("Must provide parameters file and cross-fold directory")
    else:
        params_file = args[0]
        exp_dir = args[1]

    #######################################################
    # prep work

    # count folds
    if options.num_folds is None:
        options.num_folds = 0
        fold0_dir = "%s/f%dc0" % (exp_dir, options.num_folds)
        model_file = "%s/train/model_best.h5" % fold0_dir
        if options.data_head is not None:
            model_file = "%s/train/model%d_best.h5" % (fold0_dir, options.data_head)
        while os.path.isfile(model_file):
            options.num_folds += 1
            fold0_dir = "%s/f%dc0" % (exp_dir, options.num_folds)
            model_file = "%s/train/model_best.h5" % fold0_dir
            if options.data_head is not None:
                model_file = "%s/train/model%d_best.h5" % (fold0_dir, options.data_head)
        print("Found %d folds" % options.num_folds)
        if options.num_folds == 0:
            exit(1)
    
    # subset folds
    fold_index = [fold_i for fold_i in range(options.num_folds)]

    # subset folds (list)
    if options.fold_subset_list is not None:
        fold_index = [int(fold_str) for fold_str in options.fold_subset_list.split(",")]

    # extract output subdirectory name
    gtex_out_dir = options.out_dir

    # split SNP stats
    snp_stats = options.snp_stats.split(",")

    # merge study/tissue variants
    mpos_vcf_file = "%s/pos_merge.vcf" % options.gtex_vcf_dir
    mneg_vcf_file = "%s/neg_merge.vcf" % options.gtex_vcf_dir

    ################################################################
    # SED

    # SED command base
    cmd_base = ('. %s; ' % os.environ['BASKERVILLE_CONDA']) if 'BASKERVILLE_CONDA' in os.environ else ''
    cmd_base += "conda activate %s;" % options.conda_env
    cmd_base += " echo $HOSTNAME;"

    jobs = []

    for ci in range(options.crosses):
        for fi in fold_index:
            it_dir = "%s/f%dc%d" % (exp_dir, fi, ci)
            name = "%s-f%dc%d" % (options.name, fi, ci)

            # update output directory
            it_out_dir = "%s/%s" % (it_dir, gtex_out_dir)
            os.makedirs(it_out_dir, exist_ok=True)

            # choose model
            model_file = "%s/train/model_best.h5" % it_dir
            if options.data_head is not None:
                model_file = "%s/train/model%d_best.h5" % (it_dir, options.data_head)

            ########################################
            # negative jobs

            # pickle options
            options.out_dir = "%s/merge_neg" % it_out_dir
            os.makedirs(options.out_dir, exist_ok=True)
            options_pkl_file = "%s/options.pkl" % options.out_dir
            options_pkl = open(options_pkl_file, "wb")
            pickle.dump(options, options_pkl)
            options_pkl.close()

            # create base fold command
            cmd_fold = "%s hound_snpgene.py %s %s %s" % (
                cmd_base,
                options_pkl_file,
                params_file,
                model_file,
            )

            for pi in range(options.processes):
                scores_file = "%s/job%d/scores.h5" % (options.out_dir, pi)
                if not nonzero_h5(scores_file, snp_stats):
                    cmd_job = "%s %s %d" % (cmd_fold, mneg_vcf_file, pi)
                    j = slurm.Job(
                        cmd_job,
                        "%s_neg%d" % (name, pi),
                        "%s/job%d.out" % (options.out_dir, pi),
                        "%s/job%d.err" % (options.out_dir, pi),
                        "%s/job%d.sb" % (options.out_dir, pi),
                        queue=options.queue,
                        gpu=1,
                        cpu=2,
                        mem=30000,
                        time="7-0:0:0",
                    )
                    jobs.append(j)

            ########################################
            # positive jobs

            # pickle options
            options.out_dir = "%s/merge_pos" % it_out_dir
            os.makedirs(options.out_dir, exist_ok=True)
            options_pkl_file = "%s/options.pkl" % options.out_dir
            options_pkl = open(options_pkl_file, "wb")
            pickle.dump(options, options_pkl)
            options_pkl.close()

            # create base fold command
            cmd_fold = "%s hound_snpgene.py %s %s %s" % (
                cmd_base,
                options_pkl_file,
                params_file,
                model_file,
            )

            for pi in range(options.processes):
                scores_file = "%s/job%d/scores.h5" % (options.out_dir, pi)
                if not nonzero_h5(scores_file, snp_stats):
                    cmd_job = "%s %s %d" % (cmd_fold, mpos_vcf_file, pi)
                    j = slurm.Job(
                        cmd_job,
                        "%s_pos%d" % (name, pi),
                        "%s/job%d.out" % (options.out_dir, pi),
                        "%s/job%d.err" % (options.out_dir, pi),
                        "%s/job%d.sb" % (options.out_dir, pi),
                        queue=options.queue,
                        gpu=1,
                        cpu=2,
                        mem=30000,
                        time="7-0:0:0",
                    )
                    jobs.append(j)

    slurm.multi_run(
        jobs, max_proc=options.max_proc, verbose=True, launch_sleep=10, update_sleep=60
    )

    #######################################################
    # collect output

    for ci in range(options.crosses):
        for fi in fold_index:
            it_out_dir = "%s/f%dc%d/%s" % (exp_dir, fi, ci, gtex_out_dir)

            # collect negatives
            neg_out_dir = "%s/merge_neg" % it_out_dir
            if not os.path.isfile("%s/scores.h5" % neg_out_dir):
                print(f"Collecting {neg_out_dir}")
                collect_scores(neg_out_dir, options.processes, "scores.h5")

            # collect positives
            pos_out_dir = "%s/merge_pos" % it_out_dir
            if not os.path.isfile("%s/scores.h5" % pos_out_dir):
                print(f"Collecting {pos_out_dir}")
                collect_scores(pos_out_dir, options.processes, "scores.h5")

    ################################################################
    # split study/tissue variants

    for ci in range(options.crosses):
        for fi in fold_index:
            it_out_dir = "%s/f%dc%d/%s" % (exp_dir, fi, ci, gtex_out_dir)
            print(f"Splitting {it_out_dir}")

            # split positives
            split_scores(it_out_dir, "pos", options.gtex_vcf_dir, snp_stats)

            # split negatives
            split_scores(it_out_dir, "neg", options.gtex_vcf_dir, snp_stats)

    ################################################################
    # ensemble

    ensemble_dir = "%s/ensemble" % exp_dir
    if not os.path.isdir(ensemble_dir):
        os.mkdir(ensemble_dir)

    gtex_dir = "%s/%s" % (ensemble_dir, gtex_out_dir)
    if not os.path.isdir(gtex_dir):
        os.mkdir(gtex_dir)

    for gtex_pos_vcf in glob.glob("%s/*_pos.vcf" % options.gtex_vcf_dir):
        print(f"Ensembling {gtex_pos_vcf}")
        gtex_neg_vcf = gtex_pos_vcf.replace("_pos.", "_neg.")
        pos_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
        neg_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]

        # collect score files
        score_pos_files = []
        score_neg_files = []
        for ci in range(options.crosses):
            for fi in fold_index:
                it_dir = "%s/f%dc%d" % (exp_dir, fi, ci)
                it_out_dir = "%s/%s" % (it_dir, gtex_out_dir)

                score_pos_file = "%s/%s/scores.h5" % (it_out_dir, pos_base)
                score_pos_files.append(score_pos_file)

                score_neg_file = "%s/%s/scores.h5" % (it_out_dir, neg_base)
                score_neg_files.append(score_neg_file)

        # ensemble
        ens_pos_dir = "%s/%s" % (gtex_dir, pos_base)
        os.makedirs(ens_pos_dir, exist_ok=True)
        ens_pos_file = "%s/scores.h5" % (ens_pos_dir)
        if not os.path.isfile(ens_pos_file):
            ensemble_h5(ens_pos_file, score_pos_files, snp_stats)

        ens_neg_dir = "%s/%s" % (gtex_dir, neg_base)
        os.makedirs(ens_neg_dir, exist_ok=True)
        ens_neg_file = "%s/scores.h5" % (ens_neg_dir)
        if not os.path.isfile(ens_neg_file):
            ensemble_h5(ens_neg_file, score_neg_files, snp_stats)

    ################################################################
    # coefficient analysis

    cmd_base = "westminster_gtexg_coef.py -g %s" % options.gtex_vcf_dir

    jobs = []
    for ci in range(options.crosses):
        for fi in fold_index:
            it_dir = "%s/f%dc%d" % (exp_dir, fi, ci)
            it_out_dir = "%s/%s" % (it_dir, gtex_out_dir)

            for snp_stat in snp_stats:
                coef_out_dir = f"{it_out_dir}/coef-{snp_stat}"
                cmd_coef = f"{cmd_base} -o {coef_out_dir} -s {snp_stat} {it_out_dir}"
                j = slurm.Job(
                    cmd_coef,
                    "coef",
                    f"{coef_out_dir}.out",
                    f"{coef_out_dir}.err",
                    queue="standard",
                    cpu=2,
                    mem=22000,
                    time="12:0:0",
                )
                jobs.append(j)

    # ensemble
    it_out_dir = f"{exp_dir}/ensemble/{gtex_out_dir}"
    for snp_stat in snp_stats:
        coef_out_dir = f"{it_out_dir}/coef-{snp_stat}"
        cmd_coef = f"{cmd_base} -o {coef_out_dir} -s {snp_stat} {it_out_dir}"
        j = slurm.Job(
            cmd_coef,
            "coef",
            f"{coef_out_dir}.out",
            f"{coef_out_dir}.err",
            queue="standard",
            cpu=2,
            mem=22000,
            time="12:0:0",
        )
        jobs.append(j)

    slurm.multi_run(jobs, verbose=True)


def collect_scores(out_dir: str, num_jobs: int, h5f_name: str = "scores.h5"):
    """Collect parallel SAD jobs' output into one HDF5.

    Args:
      out_dir (str): Output directory.
      num_jobs (int): Number of jobs to combine results from.
    """
    # count variants
    num_variants = 0
    for pi in range(num_jobs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, h5f_name)
        with h5py.File(job_h5_file, "r") as job_h5_open:
            num_variants += len(job_h5_open["snp"])

    final_dict = {}

    for pi in range(num_jobs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, h5f_name)
        with h5py.File(job_h5_file, "r") as job_h5_open:
            for key in job_h5_open.keys():
                if key in ["target_ids", "target_labels"]:
                    final_dict[key] = job_h5_open[key][:]
                elif key in ["snp", "chr", "pos", "ref_allele", "alt_allele", "gene"]:
                    final_dict.setdefault(key, []).append(job_h5_open[key][:])
                elif isinstance(job_h5_open[key], h5py.Group):
                    snp_stat = key
                    if snp_stat not in final_dict:
                        final_dict[snp_stat] = {}
                    for snp in job_h5_open[snp_stat].keys():
                        final_dict[snp_stat][snp] = {}
                        for gene in job_h5_open[snp_stat][snp].keys():
                            final_dict[snp_stat][snp][gene] = job_h5_open[snp_stat][
                                snp
                            ][gene][:]
                else:
                    print(f"During collection, unknown key {key}")

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, h5f_name)
    with h5py.File(final_h5_file, "w") as final_h5_open:
        for key in final_dict.keys():
            if key in ["target_ids", "target_labels"]:
                final_h5_open.create_dataset(key, data=final_dict[key])
            elif key in ["snp", "chr", "pos", "ref_allele", "alt_allele", "gene"]:
                fdv = np.concatenate(final_dict[key])
                final_h5_open.create_dataset(key, data=fdv)

            else:
                snp_stat = key
                final_h5_open.create_group(snp_stat)
                for snp in final_dict[snp_stat].keys():
                    final_h5_open[snp_stat].create_group(snp)
                    for gene in final_dict[snp_stat][snp].keys():
                        final_h5_open[snp_stat][snp].create_dataset(
                            gene, data=final_dict[snp_stat][snp][gene]
                        )


def ensemble_h5(ensemble_h5_file: str, scores_files: list, snp_stats: list):
    """Ensemble scores from multiple files into a single file.

    Args:
      ensemble_h5_file (str): ensemble score HDF5.
      scores_files ([str]): list of replicate score HDFs.
      snp_stats ([str]): SNP stats to average over folds.
    """
    # open ensemble
    ensemble_h5 = h5py.File(ensemble_h5_file, "w")

    with h5py.File(scores_files[0], "r") as scores0_h5:
        for key in scores0_h5.keys():
            if key in snp_stats:
                ensemble_h5.create_group(key)
            else:
                ensemble_h5.create_dataset(key, data=scores0_h5[key])

    # average stats
    num_folds = len(scores_files)
    for snp_stat in snp_stats:
        # sum scores across folds
        snpgene_scores = {}
        for scores_file in scores_files:
            with h5py.File(scores_file, "r") as scores_h5:
                for snp in scores_h5[snp_stat].keys():
                    if snp not in snpgene_scores:
                        snpgene_scores[snp] = {}
                    for gene in scores_h5[snp_stat][snp].keys():
                        if gene not in snpgene_scores[snp]:
                            snpgene_scores[snp][gene] = scores_h5[snp_stat][snp][gene][
                                :
                            ].astype("float32")
                        else:
                            snpgene_scores[snp][gene] += scores_h5[snp_stat][snp][gene][
                                :
                            ].astype("float32")

        # write average score
        for snp in snpgene_scores:
            ensemble_h5[snp_stat].create_group(snp)
            for gene in snpgene_scores[snp]:
                ensemble_score = snpgene_scores[snp][gene] / num_folds
                ensemble_h5[snp_stat][snp].create_dataset(
                    gene, data=ensemble_score.astype("float16")
                )

    ensemble_h5.close()


def nonzero_h5(h5_file: str, stat_keys):
    """Verify the HDF5 exists, and there are nonzero values
      for each stat key given.

    Args:
      h5_file (str): HDF5 file name.
      stat_keys ([str]): List of SNP stat keys.
    """
    if os.path.isfile(h5_file):
        try:
            with h5py.File(h5_file, "r") as h5_open:
                snps_all = set([snp.decode("UTF-8") for snp in h5_open["snp"]])
                for sk in stat_keys:
                    snps_stat = set(h5_open[sk].keys())
                    snps_ovl = snps_all & snps_stat
                    if len(snps_ovl) == 0:
                        print(f"{h5_file}: {sk} empty.")
                        return False
                    else:
                        for snp in list(snps_ovl)[:5]:
                            for gene in h5_open[sk][snp].keys():
                                score = h5_open[sk][snp][gene][:]
                                if score.var(dtype="float64") == 0:
                                    print(f"{h5_file}: {sk} {snp} {gene} zero var.")
                                    return False
                return True
        except:
            print(f"{h5_file}: error", sys.exc_info()[0])
            return False
    else:
        return False


def split_scores(it_out_dir: str, posneg: str, vcf_dir: str, snp_stats):
    """Split merged VCF predictions in HDF5 into tissue-specific
    predictions in HDF5.

    Args:
      it_out_dir (str): output directory for iteration.
      posneg (str): 'pos' or 'neg'.
      vcf_dir (str): directory containing tissue-specific VCFs.
      snp_stats ([str]]): list of SED stats.
    """
    merge_h5_file = "%s/merge_%s/scores.h5" % (it_out_dir, posneg)
    merge_h5 = h5py.File(merge_h5_file, "r")

    # hash scored SNPs
    all_snps = set([snp.decode("UTF-8") for snp in merge_h5["snp"]])
    scored_snps = set([snp for snp in merge_h5[snp_stats[0]].keys()])

    # for each tissue VCF
    vcf_glob = "%s/*_%s.vcf" % (vcf_dir, posneg)
    for tissue_vcf_file in glob.glob(vcf_glob):
        tissue_label = tissue_vcf_file.split("/")[-1]
        tissue_label = tissue_label.replace("_pos.vcf", "")
        tissue_label = tissue_label.replace("_neg.vcf", "")

        # initialize HDF5 arrays
        snpg_snp = []
        snpg_chr = []
        snpg_pos = []
        snpg_ref = []
        snpg_alt = []

        # fill HDF5 arrays with ordered SNPs
        for line in open(tissue_vcf_file):
            if not line.startswith("#"):
                a = line.split()
                chrm, pos, snp, ref, alt = a[:5]

                # SNPs w/o genes disappear
                if snp in all_snps:
                    snpg_snp.append(snp)
                    snpg_chr.append(chrm)
                    snpg_pos.append(int(pos))
                    snpg_ref.append(ref)
                    snpg_alt.append(alt)

        # write tissue HDF5
        tissue_dir = "%s/%s_%s" % (it_out_dir, tissue_label, posneg)
        os.makedirs(tissue_dir, exist_ok=True)
        with h5py.File("%s/scores.h5" % tissue_dir, "w") as tissue_h5:
            # write SNPs
            tissue_h5.create_dataset("snp", data=np.array(snpg_snp, "S"))

            # write chr
            tissue_h5.create_dataset("chr", data=np.array(snpg_chr, "S"))

            # write SNP pos
            tissue_h5.create_dataset("pos", data=np.array(snpg_pos, dtype="uint32"))

            # write ref allele
            tissue_h5.create_dataset("ref_allele", data=np.array(snpg_ref, dtype="S"))

            # write alt allele
            tissue_h5.create_dataset("alt_allele", data=np.array(snpg_alt, dtype="S"))

            # write targets
            tissue_h5.create_dataset("target_ids", data=merge_h5["target_ids"])
            tissue_h5.create_dataset("target_labels", data=merge_h5["target_labels"])

            # write SNP stats
            genes = set()
            for ss in snp_stats:
                tissue_h5.create_group(ss)
                for snp in snpg_snp:
                    if snp in scored_snps:
                        tissue_h5[ss].create_group(snp)
                        for gene in merge_h5[ss][snp].keys():
                            tissue_h5[ss][snp].create_dataset(
                                gene, data=merge_h5[ss][snp][gene][:]
                            )
                            genes.add(gene)

            # write genes
            tissue_h5.create_dataset("gene", data=np.array(sorted(genes), "S"))

    merge_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
