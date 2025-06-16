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

import h5py
import numpy as np

import slurm

from westminster.multi import collect_scores, nonzero_h5

"""
westminster_gnomad_folds.py

Benchmark Baskerville model replicates on Gnomad common vs rare variant classification.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <data_dir>"
    parser = OptionParser(usage)

    # snp options
    snp_options = OptionGroup(parser, "hound_snp.py options")
    parser.add_option(
        "-c",
        dest="cluster_snps_pct",
        default=0,
        type="float",
        help="Cluster SNPs within a %% of the seq length to make a single ref pred [Default: %default]",
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
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts [Default: %default]",
    )
    snp_options.add_option(
        "-o",
        dest="out_dir",
        default="gtex",
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
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    parser.add_option_group(snp_options)

    # classify
    class_options = OptionGroup(parser, "westminster_classify.py options")
    class_options.add_option(
        "--cn",
        dest="class_name",
        default=None,
        help="Classifier name extension [Default: %default]",
    )
    class_options.add_option(
        "--ct",
        dest="class_targets_file",
        default=None,
        help="Targets slice for the classifier stage [Default: %default]",
    )
    class_options.add_option(
        "-l",
        dest="learning_rate",
        default=0.05,
        type="float",
        help="XGBoost learning rate [Default: %default]",
    )
    class_options.add_option(
        "-n",
        dest="n_estimators",
        default=100,
        type="int",
        help="XGBoost n_estimators [Default: %default]",
    )
    class_options.add_option(
        "--md",
        dest="max_depth",
        default=4,
        type="int",
        help="XGBoost max_depth [Default: %default]",
    )
    parser.add_option_group(class_options)

    # cross-fold
    fold_options = OptionGroup(parser, "cross-fold options")
    fold_options.add_option(
        "--cross",
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
        default="tf15",
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
        "-g",
        "--gnomad",
        dest="gnomad_vcf_dir",
        default="/group/fdna/public/genomes/hg38/gnomad",
    )
    fold_options.add_option(
        "--local",
        dest="local",
        default=False,
        action="store_true",
        help="Run locally [Default: %default]",
    )
    fold_options.add_option(
        "--name",
        dest="name",
        default="gnomad",
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
    fold_options.add_option(
        "-v",
        dest="variants_label",
        default="31k",
        help="Variant number label [Default: %default]",
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
        fold0_dir = f"{exp_dir}/f0c0"
        model_file = f"{fold0_dir}/train/model_best.h5"
        if options.data_head is not None:
            model_file = f"{fold0_dir}/train/model{options.data_head}_best.h5"
        while os.path.isfile(model_file):
            options.num_folds += 1
            fold0_dir = f"{exp_dir}/f{options.num_folds}c0"
            model_file = f"{fold0_dir}/train/model_best.h5"
            if options.data_head is not None:
                model_file = f"{fold0_dir}/train/model{options.data_head}_best.h5"
        print(f"Found {options.num_folds} folds")
        if options.num_folds == 0:
            exit(1)

    # extract output subdirectory name
    gnomad_out_dir = options.out_dir

    # split SNP stats
    snp_stats = options.snp_stats.split(",")

    # merge study/tissue variants
    rare_vcf_file = f"{options.gnomad_vcf_dir}/rare{options.variants_label}.vcf"
    common_vcf_file = f"{options.gnomad_vcf_dir}/common{options.variants_label}.vcf"

    ################################################################
    # SAD

    # SAD command base
    cmd_base = ". /home/drk/anaconda3/etc/profile.d/conda.sh;"
    cmd_base += f" conda activate {options.conda_env};"
    cmd_base += " echo $HOSTNAME;"

    jobs = []

    for ci in range(options.crosses):
        for fi in range(options.num_folds):
            it_dir = f"{exp_dir}/f{fi}c{ci}"
            name = f"{options.name}-f{fi}c{ci}"

            # update output directory
            it_out_dir = f"{it_dir}/{gnomad_out_dir}"
            os.makedirs(it_out_dir, exist_ok=True)

            # choose model
            model_file = f"{it_dir}/train/model_best.h5"
            if options.data_head is not None:
                model_file = f"{it_dir}/train/model{options.data_head}_best.h5"

            ########################################
            # rare jobs

            # pickle options
            options.out_dir = f"{it_out_dir}/rare{options.variants_label}"
            os.makedirs(options.out_dir, exist_ok=True)
            options_pkl_file = "%s/options.pkl" % options.out_dir
            options_pkl = open(options_pkl_file, "wb")
            pickle.dump(options, options_pkl)
            options_pkl.close()

            # create base fold command
            cmd_fold = "%s time hound_snp.py %s %s %s" % (
                cmd_base,
                options_pkl_file,
                params_file,
                model_file,
            )

            for pi in range(options.processes):
                scores_file = f"{options.out_dir}/job{pi}/scores.h5"
                if not nonzero_h5(scores_file, snp_stats):
                    cmd_job = "%s %s %d" % (cmd_fold, rare_vcf_file, pi)
                    j = slurm.Job(
                        cmd_job,
                        f"{name}_rare{pi}",
                        f"{options.out_dir}/job{pi}.out",
                        f"{options.out_dir}/job{pi}.err",
                        f"{options.out_dir}/job{pi}.sb",
                        queue=options.queue,
                        gpu=1,
                        cpu=4,
                        mem=30000,
                        time="7-0:0:0",
                    )
                    jobs.append(j)

            ########################################
            # common jobs

            # pickle options
            options.out_dir = f"{it_out_dir}/common{options.variants_label}"
            os.makedirs(options.out_dir, exist_ok=True)
            options_pkl_file = "%s/options.pkl" % options.out_dir
            options_pkl = open(options_pkl_file, "wb")
            pickle.dump(options, options_pkl)
            options_pkl.close()

            # create base fold command
            cmd_fold = "%s time hound_snp.py %s %s %s" % (
                cmd_base,
                options_pkl_file,
                params_file,
                model_file,
            )

            for pi in range(options.processes):
                scores_file = f"{options.out_dir}/job{pi}/scores.h5"
                if not nonzero_h5(scores_file, snp_stats):
                    cmd_job = "%s %s %d" % (cmd_fold, common_vcf_file, pi)
                    j = slurm.Job(
                        cmd_job,
                        f"{name}_common{pi}",
                        f"{options.out_dir}/job{pi}.out",
                        f"{options.out_dir}/job{pi}.err",
                        f"{options.out_dir}/job{pi}.sb",
                        queue=options.queue,
                        gpu=1,
                        cpu=4,
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
        for fi in range(options.num_folds):
            it_out_dir = f"{exp_dir}/f{fi}c{ci}/{gnomad_out_dir}"

            # collect rare
            rare_out_dir = f"{it_out_dir}/rare{options.variants_label}"
            if not os.path.isfile(f"{rare_out_dir}/scores.h5"):
                collect_scores(rare_out_dir, options.processes)

            # collect positives
            common_out_dir = f"{it_out_dir}/common{options.variants_label}"
            if not os.path.isfile(f"{common_out_dir}/scores.h5"):
                collect_scores(common_out_dir, options.processes)

    ################################################################
    # ensemble

    ensemble_dir = f"{exp_dir}/ensemble"
    gnomad_dir = f"{ensemble_dir}/{gnomad_out_dir}"
    os.makedirs(gnomad_dir, exist_ok=True)

    # collect scores
    scores_rare_files = []
    scores_common_files = []
    for ci in range(options.crosses):
        for fi in range(options.num_folds):
            it_dir = f"{exp_dir}/f{fi}c{ci}"
            it_out_dir = f"{it_dir}/{gnomad_out_dir}"

            scores_rare_file = f"{it_out_dir}/rare{options.variants_label}/scores.h5"
            scores_rare_files.append(scores_rare_file)

            scores_common_file = (
                f"{it_out_dir}/common{options.variants_label}/scores.h5"
            )
            scores_common_files.append(scores_common_file)

    # ensemble
    ens_rare_dir = f"{gnomad_dir}/rare{options.variants_label}"
    os.makedirs(ens_rare_dir, exist_ok=True)
    ens_rare_file = f"{ens_rare_dir}/scores.h5"
    if not os.path.isfile(ens_rare_file):
        ensemble_scores(ens_rare_file, scores_rare_files)

    ens_common_dir = f"{gnomad_dir}/common{options.variants_label}"
    os.makedirs(ens_common_dir, exist_ok=True)
    ens_common_file = f"{ens_common_dir}/scores.h5"
    if not os.path.isfile(ens_common_file):
        ensemble_scores(ens_common_file, scores_common_files)

    ################################################################
    # fit classifiers

    cmd_base = "westminster_classify.py -f 10 -i 10 -x"
    cmd_base += f" -l {options.learning_rate}"
    cmd_base += f" --md {options.max_depth}"
    cmd_base += f" -n {options.n_estimators}"

    if options.class_targets_file is not None:
        cmd_base += " -t %s" % options.class_targets_file

    classify_stats = snp_stats
    if len(classify_stats) > 1:
        classify_stats.append(options.snp_stats)

    jobs = []
    for ci in range(options.crosses):
        for fi in range(1, options.num_folds):
            it_dir = f"{exp_dir}/f{fi}c{ci}"
            it_out_dir = f"{it_dir}/{gnomad_out_dir}"

            for snp_stat in classify_stats:
                stat_label = snp_stat.replace(",", "-")
                class_out_dir = (
                    f"{it_out_dir}/class{options.variants_label}-{stat_label}"
                )
                if options.class_name is not None:
                    class_out_dir += f"-{options.class_name}"

                if not os.path.isfile(f"{class_out_dir}/stats.txt"):
                    scores_rare_file = (
                        f"{it_out_dir}/rare{options.variants_label}/scores.h5"
                    )
                    scores_common_file = (
                        f"{it_out_dir}/common{options.variants_label}/scores.h5"
                    )

                    cmd_class = f"{cmd_base} -o {class_out_dir}"
                    cmd_class += f" --stat {snp_stat}"
                    cmd_class += f" {scores_rare_file} {scores_common_file}"

                    # TEMP
                    j = slurm.Job(
                        cmd_class,
                        "gnomad",
                        f"{class_out_dir}.out",
                        f"{class_out_dir}.err",
                        f"{class_out_dir}.sb",
                        queue="standard",
                        cpu=16,
                        mem=240000,
                        time="3-0:0:0",
                    )
                    jobs.append(j)

    # ensemble
    it_out_dir = f"{exp_dir}/ensemble/{gnomad_out_dir}"
    for snp_stat in classify_stats:
        stat_label = snp_stat.replace(",", "-")
        class_out_dir = f"{it_out_dir}/class{options.variants_label}-{stat_label}"
        if options.class_name is not None:
            class_out_dir += f"-{options.class_name}"

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
                mem=240000,
                time="3-0:0:0",
            )
            jobs.append(j)

    slurm.multi_run(jobs, verbose=True)


def ensemble_scores(ensemble_h5_file: str, scores_files):
    """Ensemble scores from multiple files into a single file.

    Args:
      ensemble_h5_file (str): ensemble score HDF5.
      scores_files ([str]): list of replicate score HDFs.
    """
    # open ensemble
    ensemble_h5 = h5py.File(ensemble_h5_file, "w")

    # transfer base
    base_keys = [
        "alt_allele",
        "chr",
        "pos",
        "ref_allele",
        "snp",
        "target_ids",
        "target_labels",
    ]
    snp_stats = []
    sad_shapes = []
    scores0_h5 = h5py.File(scores_files[0], "r")
    for key in scores0_h5.keys():
        if key in base_keys:
            ensemble_h5.create_dataset(key, data=scores0_h5[key])
        else:
            snp_stats.append(key)
            sad_shapes.append(scores0_h5[key].shape)
    scores0_h5.close()

    # average stats
    num_folds = len(scores_files)
    for si, snp_stat in enumerate(snp_stats):
        # initialize ensemble array
        sad_values = np.zeros(shape=sad_shapes[si], dtype="float32")

        # read and add folds
        for scores_file in scores_files:
            with h5py.File(scores_file, "r") as scores_h5:
                sad_values += scores_h5[snp_stat][:].astype("float32")

        # normalize and downcast
        sad_values /= num_folds
        sad_values = sad_values.astype("float16")

        # save
        ensemble_h5.create_dataset(snp_stat, data=sad_values)

    ensemble_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()