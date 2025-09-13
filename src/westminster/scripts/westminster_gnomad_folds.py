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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import slurm

from baskerville_torch import utils
from baskerville_torch.scripts.hound_snp_folds import snp_folds

"""
westminster_gnomad_folds.py

Benchmark Baskerville model replicates on Gnomad common vs rare variant classification.
"""


################################################################################
# main
################################################################################
def main():
    parser = ArgumentParser(
        description="Compute variant effect predictions for SNPs in a VCF file using cross-fold model ensemble.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # snp options
    snp_group = parser.add_argument_group("hound_snp.py options")
    snp_group.add_argument(
        "-c",
        dest="cluster_pct",
        default=0,
        type=float,
        help="Cluster SNPs (or genes) within a %% of the seq length to make a single ref pred",
    )
    snp_group.add_argument(
        "-f",
        dest="genome_fasta",
        default=f"{os.environ['HG38']}/assembly/ucsc/hg38.fa",
        help="Genome FASTA for sequences",
    )
    snp_group.add_argument(
        "--head",
        dest="head",
        default=0,
        type=int,
        help="Model head with which to predict.",
    )
    snp_group.add_argument(
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts",
    )
    snp_group.add_argument(
        "-n",
        "--norm",
        dest="norm_subdir",
        default=None,
        help="Model directory subdirectory containing normalization HDF5 files for each fold",
    )
    snp_group.add_argument(
        "-o",
        dest="out_dir",
        default="gnomad",
        help="Output directory for tables and plots",
    )
    snp_group.add_argument(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions",
    )
    snp_group.add_argument(
        "--shifts",
        dest="shifts",
        default="0",
        type=str,
        help="Ensemble prediction shifts",
    )
    snp_group.add_argument(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save.",
    )
    snp_group.add_argument(
        "-t",
        dest="targets_file",
        default=None,
        type=str,
        help="File specifying target indexes and labels in table format",
    )

    # classify options
    class_group = parser.add_argument_group("westminster_classify.py options")
    class_group.add_argument(
        "--cn",
        dest="class_name",
        default=None,
        help="Classifier name extension",
    )
    class_group.add_argument(
        "--ct",
        dest="class_targets_file",
        default=None,
        help="Targets slice for the classifier stage",
    )
    class_group.add_argument(
        "-l",
        dest="learning_rate",
        default=0.05,
        type=float,
        help="XGBoost learning rate",
    )
    class_group.add_argument(
        "--md",
        dest="max_depth",
        default=4,
        type=int,
        help="XGBoost max_depth",
    )
    class_group.add_argument(
        "--ne",
        dest="n_estimators",
        default=100,
        type=int,
        help="XGBoost n_estimators",
    )

    # cross-fold options
    fold_group = parser.add_argument_group("cross-fold options")
    fold_group.add_argument(
        "--cross",
        dest="crosses",
        default=1,
        type=int,
        help="Number of cross-fold rounds",
    )
    fold_group.add_argument(
        "-e", dest="conda_env", default="torch2.6", help="Anaconda environment"
    )
    fold_group.add_argument(
        "--embed",
        default=False,
        action="store_true",
        help="Embed output in the models directory",
    )
    fold_group.add_argument(
        "--folds",
        dest="num_folds",
        default=None,
        type=int,
        help="Number of folds to evaluate",
    )
    fold_group.add_argument(
        "--f_list",
        dest="fold_subset_list",
        default=None,
        help="Subset of folds to evaluate (encoded as comma-separated string)",
    )
    fold_group.add_argument(
        "-g",
        "--gnomad",
        dest="gnomad_vcf_dir",
        default="/group/fdna/public/genomes/hg38/gnomad",
        help="Directory with GnomAD VCF files",
    )
    fold_group.add_argument(
        "--local", dest="local", default=False, action="store_true", help="Run locally"
    )
    fold_group.add_argument(
        "--name", dest="name", default="gnomad", help="SLURM name prefix"
    )
    fold_group.add_argument(
        "-p",
        "--parallel_jobs",
        dest="parallel_jobs",
        default=None,
        type=int,
        help="Maximum number of jobs to run in parallel",
    )
    fold_group.add_argument(
        "-j",
        dest="job_size",
        default=256,
        type=int,
        help="Number of SNPs to process per job",
    )
    fold_group.add_argument(
        "-q",
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs",
    )
    fold_group.add_argument(
        "-v",
        dest="variants_label",
        default="31k",
        help="Variant number label",
    )

    # Positional arguments
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("models_dir", help="Cross-fold models directory")
    args = parser.parse_args()

    #######################################################
    # prep work

    # count folds
    if args.num_folds is None:
        args.num_folds = utils.detect_model_folds(args.models_dir)
        print(f"Found {args.num_folds} folds")
        if args.num_folds == 0:
            raise ValueError(f"No models found in {args.models_dir}")

    # extract output subdirectory name
    gnomad_out_dir = args.out_dir

    # split SNP stats
    snp_stats = args.snp_stats.split(",")

    ################################################################
    # score SNPs

    # merge study/tissue variants
    rare_vcf_file = f"{args.gnomad_vcf_dir}/rare{args.variants_label}.vcf"
    common_vcf_file = f"{args.gnomad_vcf_dir}/common{args.variants_label}.vcf"

    # embed output in the models directory
    args.embed = True

    # score rare SNPs
    args.vcf_file = rare_vcf_file
    args.out_dir = f"{gnomad_out_dir}/rare{args.variants_label}"
    snp_folds(args)

    # score common SNPs
    args.vcf_file = common_vcf_file
    args.out_dir = f"{gnomad_out_dir}/common{args.variants_label}"
    snp_folds(args)

    ################################################################
    # fit classifiers

    cmd_base = "westminster_classify.py -f 10 -i 10 -x"
    cmd_base += f" -l {args.learning_rate}"
    cmd_base += f" --md {args.max_depth}"
    cmd_base += f" --ne {args.n_estimators}"

    if args.class_targets_file is not None:
        cmd_base += f" -t {args.class_targets_file}"

    classify_stats = snp_stats
    if len(classify_stats) > 1:
        classify_stats.append(args.snp_stats)

    jobs = []
    for ci in range(args.crosses):
        for fi in range(args.num_folds):
            it_dir = f"{args.models_dir}/f{fi}c{ci}"
            it_out_dir = f"{it_dir}/{gnomad_out_dir}"

            for snp_stat in classify_stats:
                stat_label = snp_stat.replace(",", "-")
                class_out_dir = f"{it_out_dir}/class{args.variants_label}-{stat_label}"
                if args.class_name is not None:
                    class_out_dir += f"-{args.class_name}"

                if not os.path.isfile(f"{class_out_dir}/stats.txt"):
                    scores_rare_file = (
                        f"{it_out_dir}/rare{args.variants_label}/scores.h5"
                    )
                    scores_common_file = (
                        f"{it_out_dir}/common{args.variants_label}/scores.h5"
                    )

                    cmd_class = f"{cmd_base} -o {class_out_dir}"
                    cmd_class += f" --stat {snp_stat}"
                    cmd_class += f" {scores_rare_file} {scores_common_file}"

                    j = slurm.Job(
                        cmd_class,
                        "gnomad",
                        f"{class_out_dir}.out",
                        f"{class_out_dir}.err",
                        f"{class_out_dir}.sb",
                        queue="standard",
                        cpu=16,
                        mem=60000,
                        time="1-0:0:0",
                    )
                    jobs.append(j)

    # ensemble classifier jobs
    ensemble_out_dir = f"{args.models_dir}/ensemble/{gnomad_out_dir}"
    ens_rare_dir = f"{ensemble_out_dir}/rare{args.variants_label}"
    ens_common_dir = f"{ensemble_out_dir}/common{args.variants_label}"

    for snp_stat in classify_stats:
        stat_label = snp_stat.replace(",", "-")
        class_out_dir = f"{ensemble_out_dir}/class{args.variants_label}-{stat_label}"
        if args.class_name is not None:
            class_out_dir += f"-{args.class_name}"

        if not os.path.isfile(f"{class_out_dir}/stats.txt"):
            scores_rare_file = f"{ens_rare_dir}/scores.h5"
            scores_common_file = f"{ens_common_dir}/scores.h5"

            cmd_class = f"{cmd_base} -o {class_out_dir}"
            cmd_class += f" --stat {snp_stat}"
            cmd_class += f" {scores_rare_file} {scores_common_file}"

            j = slurm.Job(
                cmd_class,
                "gnomad",
                f"{class_out_dir}.out",
                f"{class_out_dir}.err",
                f"{class_out_dir}.sb",
                queue="standard",
                cpu=16,
                mem=6000,
                time="1-0:0:0",
            )
            jobs.append(j)

    slurm.multi_run(jobs, verbose=True)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
