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
import glob
import math
import pickle
import os
import sys

import h5py
import numpy as np
import pandas as pd
import shutil

import slurmrunner
from baskerville_torch import utils

"""
westminster_eqtl_folds_tf.py

Benchmark TensorFlow Baskerville model replicates on GTEx eQTL classification task.
Uses hound_snp.py for SNP scoring (TF-specific parallel job approach).
"""


################################################################################
# main
################################################################################
def main():
    parser = ArgumentParser(
        description="Compute eQTL predictions across TF Baskerville model replicates.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # hound_snp.py options
    snp_group = parser.add_argument_group("hound_snp.py options")
    snp_group.add_argument(
        "-c",
        dest="cluster_pct",
        default=0,
        type=float,
        help="Cluster SNPs within a %% of the seq length to make a single ref pred",
    )
    snp_group.add_argument(
        "-f",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ["HG38"],
        help="Genome FASTA for sequences",
    )
    snp_group.add_argument(
        "--float16",
        dest="float16",
        default=False,
        action="store_true",
        help="Use mixed float16 precision",
    )
    snp_group.add_argument(
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts",
    )
    snp_group.add_argument(
        "-o",
        dest="out_dir",
        default="snp_out",
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
    snp_group.add_argument(
        "-u", dest="untransform_old", default=False, action="store_true"
    )
    snp_group.add_argument(
        "--gcs",
        dest="gcs",
        default=False,
        action="store_true",
        help="Input and output are in gcs",
    )
    snp_group.add_argument(
        "--require_gpu",
        dest="require_gpu",
        default=False,
        action="store_true",
        help="Only run on GPU",
    )
    snp_group.add_argument(
        "--tensorrt",
        dest="tensorrt",
        default=False,
        action="store_true",
        help="Model type is tensorrt optimized",
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
        "-d",
        dest="data_head",
        default=None,
        type=int,
        help="Index for dataset/head",
    )
    fold_group.add_argument(
        "-e",
        dest="conda_env",
        default="tf15",
        help="Anaconda environment",
    )
    fold_group.add_argument(
        "--ems",
        default=False,
        action="store_true",
        help="Use legacy EMS pipeline (reads susie TSV instead of VCF INFO)",
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
        "--gtex",
        dest="gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_fine/susie_pip90r",
        help="Directory with GTEx VCF files",
    )
    fold_group.add_argument(
        "--local",
        dest="local",
        default=False,
        action="store_true",
        help="Run jobs locally instead of submitting to SLURM",
    )
    fold_group.add_argument(
        "--metrics_only",
        default=False,
        action="store_true",
        help="Skip SNP scoring and splitting; only run metrics analysis",
    )
    fold_group.add_argument(
        "--name",
        dest="name",
        default="snp",
        help="SLURM name prefix",
    )
    fold_group.add_argument(
        "-j",
        dest="job_size",
        default=512,
        type=int,
        help="Number of SNPs per scoring job",
    )
    fold_group.add_argument(
        "-p",
        dest="parallel_jobs",
        default=None,
        type=int,
        help="Maximum number of jobs to run in parallel",
    )
    fold_group.add_argument(
        "-q",
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs",
    )

    # GTEx-specific options
    gtex_group = parser.add_argument_group("GTEx options")
    gtex_group.add_argument(
        "--cn",
        dest="class_name",
        default=None,
        help="Classifier name extension",
    )
    gtex_group.add_argument(
        "--ct",
        dest="class_targets_file",
        default=None,
        help="Targets slice for the classifier stage",
    )
    gtex_group.add_argument(
        "--msl",
        dest="msl",
        default=1,
        type=int,
        help="Random forest min_samples_leaf",
    )
    gtex_group.add_argument(
        "--skip_boost",
        default=False,
        action="store_true",
        help="Skip westminster_classify.py classifier stage",
    )

    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("models_dir", help="Cross-fold models directory")
    args = parser.parse_args()

    params_file = args.params_file
    exp_dir = args.models_dir

    #######################################################
    # prep work

    # count folds
    if args.num_folds is None:
        head = 0 if args.data_head is None else args.data_head
        args.num_folds = utils.detect_model_folds(exp_dir, filename=f"model{head}_best.h5")
        print(f"Found {args.num_folds} folds")
        if args.num_folds == 0:
            raise ValueError(f"No TF models found in {exp_dir}")

    # subset folds
    fold_index = list(range(args.num_folds))
    if args.fold_subset_list is not None:
        fold_index = [int(s) for s in args.fold_subset_list.split(",")]

    # extract output subdirectory name
    gtex_out_dir = args.out_dir
    ens_out_dir = f"{exp_dir}/ensemble/{gtex_out_dir}"

    # split SNP stats
    snp_stats = args.snp_stats.split(",")

    if not args.metrics_only:
        ################################################################
        # score SNPs

        # merge study/tissue variants
        mpos_vcf_file = f"{args.gtex_vcf_dir}/pos_merge.vcf"
        mneg_vcf_file = f"{args.gtex_vcf_dir}/neg_merge.vcf"

        # compute job counts from VCF sizes
        neg_num_jobs = math.ceil(count_vcf_snps(mneg_vcf_file) / args.job_size)
        pos_num_jobs = math.ceil(count_vcf_snps(mpos_vcf_file) / args.job_size)
        print(f"Neg jobs: {neg_num_jobs}  Pos jobs: {pos_num_jobs}")

        # build base command (conda activation)
        cmd_base = (
            (". %s; " % os.environ["BASKERVILLE_CONDA"])
            if "BASKERVILLE_CONDA" in os.environ
            else ""
        )
        cmd_base += "conda activate %s;" % args.conda_env
        cmd_base += " echo $HOSTNAME;"

        jobs = []

        for ci in range(args.crosses):
            for fi in fold_index:
                it_dir = f"{exp_dir}/f{fi}c{ci}"
                name = f"{args.name}-f{fi}c{ci}"

                it_out_dir = f"{it_dir}/{gtex_out_dir}"
                os.makedirs(it_out_dir, exist_ok=True)

                head = 0 if args.data_head is None else args.data_head
                model_file = f"{it_dir}/train/model{head}_best.h5"

                ########################################
                # negative jobs

                args.out_dir = f"{it_out_dir}/merge_neg"
                args.processes = neg_num_jobs  # hound_snp.py reads this from pickle
                os.makedirs(args.out_dir, exist_ok=True)
                options_pkl_file = f"{args.out_dir}/options.pkl"
                with open(options_pkl_file, "wb") as pkl:
                    pickle.dump(args, pkl)

                cmd_fold = f"{cmd_base} hound_snp.py {options_pkl_file} {params_file} {model_file}"

                for pi in range(neg_num_jobs):
                    scores_file = f"{args.out_dir}/job{pi}/scores.h5"
                    if not nonzero_h5(scores_file, snp_stats):
                        cmd_job = f"{cmd_fold} {mneg_vcf_file} {pi}"
                        j = slurmrunner.Job(
                            cmd_job,
                            f"{name}_neg{pi}",
                            f"{args.out_dir}/job{pi}.out",
                            f"{args.out_dir}/job{pi}.err",
                            f"{args.out_dir}/job{pi}.sb",
                            queue=args.queue,
                            gpu=1,
                            cpu=2,
                            mem=30000,
                            time="7-0:0:0",
                        )
                        jobs.append(j)

                ########################################
                # positive jobs

                args.out_dir = f"{it_out_dir}/merge_pos"
                args.processes = pos_num_jobs  # hound_snp.py reads this from pickle
                os.makedirs(args.out_dir, exist_ok=True)
                options_pkl_file = f"{args.out_dir}/options.pkl"
                with open(options_pkl_file, "wb") as pkl:
                    pickle.dump(args, pkl)

                cmd_fold = f"{cmd_base} hound_snp.py {options_pkl_file} {params_file} {model_file}"

                for pi in range(pos_num_jobs):
                    scores_file = f"{args.out_dir}/job{pi}/scores.h5"
                    if not nonzero_h5(scores_file, snp_stats):
                        cmd_job = f"{cmd_fold} {mpos_vcf_file} {pi}"
                        j = slurmrunner.Job(
                            cmd_job,
                            f"{name}_pos{pi}",
                            f"{args.out_dir}/job{pi}.out",
                            f"{args.out_dir}/job{pi}.err",
                            f"{args.out_dir}/job{pi}.sb",
                            queue=args.queue,
                            gpu=1,
                            cpu=2,
                            mem=30000,
                            time="7-0:0:0",
                        )
                        jobs.append(j)

        if args.local:
            utils.exec_par([j.cmd for j in jobs], args.parallel_jobs or 1, verbose=True)
        else:
            slurmrunner.multi_run(
                jobs,
                max_proc=args.parallel_jobs,
                verbose=True,
                launch_sleep=10,
                update_sleep=60,
            )

        #######################################################
        # collect output

        for ci in range(args.crosses):
            for fi in fold_index:
                it_out_dir = f"{exp_dir}/f{fi}c{ci}/{gtex_out_dir}"

                neg_out_dir = f"{it_out_dir}/merge_neg"
                if not os.path.isfile(f"{neg_out_dir}/scores.h5"):
                    print(f"Collecting {neg_out_dir}")
                    collect_scores(neg_out_dir, neg_num_jobs)

                pos_out_dir = f"{it_out_dir}/merge_pos"
                if not os.path.isfile(f"{pos_out_dir}/scores.h5"):
                    print(f"Collecting {pos_out_dir}")
                    collect_scores(pos_out_dir, pos_num_jobs)

        ################################################################
        # split study/tissue variants

        for ci in range(args.crosses):
            for fi in fold_index:
                it_out_dir = f"{exp_dir}/f{fi}c{ci}/{gtex_out_dir}"
                print(f"Splitting {it_out_dir}")
                split_scores(it_out_dir, "pos", args.gtex_vcf_dir, snp_stats)
                split_scores(it_out_dir, "neg", args.gtex_vcf_dir, snp_stats)

        ################################################################
        # ensemble

        os.makedirs(ens_out_dir, exist_ok=True)

        for gtex_pos_vcf in glob.glob(f"{args.gtex_vcf_dir}/*_pos.vcf"):
            print(f"Ensembling {gtex_pos_vcf}")
            gtex_neg_vcf = gtex_pos_vcf.replace("_pos.", "_neg.")
            pos_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
            neg_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]

            score_pos_files = []
            score_neg_files = []
            for ci in range(args.crosses):
                for fi in fold_index:
                    it_out_dir = f"{exp_dir}/f{fi}c{ci}/{gtex_out_dir}"
                    score_pos_files.append(f"{it_out_dir}/{pos_base}/scores.h5")
                    score_neg_files.append(f"{it_out_dir}/{neg_base}/scores.h5")

            ens_pos_dir = f"{ens_out_dir}/{pos_base}"
            os.makedirs(ens_pos_dir, exist_ok=True)
            ens_pos_file = f"{ens_pos_dir}/scores.h5"
            if not os.path.isfile(ens_pos_file):
                ensemble_h5(ens_pos_file, score_pos_files, snp_stats)
                shutil.copyfile(
                    score_pos_files[0].replace("scores.h5", "targets_cov.txt"),
                    f"{ens_pos_dir}/targets_cov.txt",
                )

            ens_neg_dir = f"{ens_out_dir}/{neg_base}"
            os.makedirs(ens_neg_dir, exist_ok=True)
            ens_neg_file = f"{ens_neg_dir}/scores.h5"
            if not os.path.isfile(ens_neg_file):
                ensemble_h5(ens_neg_file, score_neg_files, snp_stats)
                shutil.copyfile(
                    score_neg_files[0].replace("scores.h5", "targets_cov.txt"),
                    f"{ens_neg_dir}/targets_cov.txt",
                )

    if not args.skip_boost:
        ################################################################
        # fit classifiers

        cmd_base = "westminster_classify.py -f 8 -i 20 -n 96 -s -x"
        cmd_base += f" --msl {args.msl}"
        if args.class_targets_file is not None:
            cmd_base += f" -t {args.class_targets_file}"

        jobs = []
        for ci in range(args.crosses):
            for fi in fold_index:
                it_out_dir = f"{exp_dir}/f{fi}c{ci}/{gtex_out_dir}"

                for gtex_pos_vcf in glob.glob(f"{args.gtex_vcf_dir}/*_pos.vcf"):
                    tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
                    sad_pos = f"{it_out_dir}/{tissue}_pos/scores.h5"
                    sad_neg = f"{it_out_dir}/{tissue}_neg/scores.h5"
                    for snp_stat in snp_stats:
                        stat_label = snp_stat.replace("/", "-")
                        class_out_dir = f"{it_out_dir}/{tissue}_class-{stat_label}"
                        if args.class_name is not None:
                            class_out_dir += f"-{args.class_name}"
                        if not os.path.isfile(f"{class_out_dir}/stats.txt"):
                            cmd_class = f"{cmd_base} -o {class_out_dir} --stat {snp_stat}"
                            cmd_class += f" {sad_pos} {sad_neg}"
                            if args.local:
                                jobs.append(cmd_class)
                            else:
                                j = slurmrunner.Job(
                                    cmd_class,
                                    tissue,
                                    f"{class_out_dir}.out",
                                    f"{class_out_dir}.err",
                                    queue="standard",
                                    cpu=2,
                                    mem=22000,
                                    time="1-0:0:0",
                                )
                                jobs.append(j)

        # ensemble
        for gtex_pos_vcf in glob.glob(f"{args.gtex_vcf_dir}/*_pos.vcf"):
            tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
            sad_pos = f"{ens_out_dir}/{tissue}_pos/scores.h5"
            sad_neg = f"{ens_out_dir}/{tissue}_neg/scores.h5"
            for snp_stat in snp_stats:
                stat_label = snp_stat.replace("/", "-")
                class_out_dir = f"{ens_out_dir}/{tissue}_class-{stat_label}"
                if args.class_name is not None:
                    class_out_dir += f"-{args.class_name}"
                if not os.path.isfile(f"{class_out_dir}/stats.txt"):
                    cmd_class = f"{cmd_base} -o {class_out_dir} --stat {snp_stat}"
                    cmd_class += f" {sad_pos} {sad_neg}"
                    if args.local:
                        jobs.append(cmd_class)
                    else:
                        j = slurmrunner.Job(
                            cmd_class,
                            tissue,
                            f"{class_out_dir}.out",
                            f"{class_out_dir}.err",
                            queue="standard",
                            cpu=2,
                            mem=22000,
                            time="1-0:0:0",
                        )
                        jobs.append(j)

        if args.local:
            utils.exec_par(jobs, 3, verbose=True)
        else:
            slurmrunner.multi_run(jobs, verbose=True)

    ################################################################
    # metrics

    cmd_base = f"westminster_eqtl_gtex.py -g {args.gtex_vcf_dir}"
    if args.ems:
        cmd_base += " --ems"

    jobs = []
    for ci in range(args.crosses):
        for fi in fold_index:
            it_out_dir = f"{exp_dir}/f{fi}c{ci}/{gtex_out_dir}"
            for snp_stat in snp_stats:
                stat_label = snp_stat.replace("/", "-")
                metrics_out_dir = f"{it_out_dir}/metrics-{stat_label}"
                if not os.path.isfile(f"{metrics_out_dir}/metrics.tsv"):
                    cmd_metrics = f"{cmd_base} -o {metrics_out_dir} -s {snp_stat} {it_out_dir}"
                    if args.local:
                        jobs.append(cmd_metrics)
                    else:
                        j = slurmrunner.Job(
                            cmd_metrics,
                            "metrics",
                            f"{metrics_out_dir}.out",
                            f"{metrics_out_dir}.err",
                            queue="standard",
                            cpu=2,
                            mem=22000,
                            time="12:0:0",
                        )
                        jobs.append(j)

    # ensemble
    for snp_stat in snp_stats:
        stat_label = snp_stat.replace("/", "-")
        metrics_out_dir = f"{ens_out_dir}/metrics-{stat_label}"
        if not os.path.isfile(f"{metrics_out_dir}/metrics.tsv"):
            cmd_metrics = f"{cmd_base} -o {metrics_out_dir} -s {snp_stat} {ens_out_dir}"
            if args.local:
                jobs.append(cmd_metrics)
            else:
                j = slurmrunner.Job(
                    cmd_metrics,
                    "metrics",
                    f"{metrics_out_dir}.out",
                    f"{metrics_out_dir}.err",
                    queue="standard",
                    cpu=2,
                    mem=22000,
                    time="12:0:0",
                )
                jobs.append(j)

    if args.local:
        utils.exec_par(jobs, 3, verbose=True)
    else:
        slurmrunner.multi_run(jobs, verbose=True)


def count_vcf_snps(vcf_file: str) -> int:
    """Count non-header lines in a VCF file."""
    return sum(1 for line in open(vcf_file) if not line.startswith("#"))


def collect_scores(out_dir: str, num_jobs: int, h5f_name: str = "scores.h5"):
    """Collect parallel hound_snp.py job outputs into one HDF5.

    Args:
        out_dir: Output directory containing job{i}/ subdirs.
        num_jobs: Number of parallel jobs to combine.
        h5f_name: HDF5 filename within each job subdir.
    """
    final = {}
    for pi in range(num_jobs):
        with h5py.File(f"{out_dir}/job{pi}/{h5f_name}", "r") as jh5:
            for key in jh5.keys():
                if key in ["target_ids", "target_labels"]:
                    final[key] = jh5[key][:]
                else:
                    final.setdefault(key, []).append(jh5[key][:])

    with h5py.File(f"{out_dir}/{h5f_name}", "w") as fh5:
        for key, val in final.items():
            if key in ["target_ids", "target_labels"]:
                fh5.create_dataset(key, data=val)
            else:
                # works for 1D (snp/chr/pos) and 2D (stat arrays: num_snps x num_targets)
                fh5.create_dataset(key, data=np.concatenate(val))


def ensemble_h5(ensemble_h5_file: str, scores_files: list, snp_stats: list):
    """Ensemble scores from multiple fold files by averaging.

    Args:
        ensemble_h5_file: Output ensemble HDF5 path.
        scores_files: List of per-fold score HDF5 paths.
        snp_stats: Stat keys to average; all other keys copied from the first file.
    """
    ens = h5py.File(ensemble_h5_file, "w")

    with h5py.File(scores_files[0], "r") as h5_0:
        for key in h5_0.keys():
            if key not in snp_stats:
                ens.create_dataset(key, data=h5_0[key])

    num_folds = len(scores_files)
    for ss in snp_stats:
        stat_sum = None
        for sf in scores_files:
            with h5py.File(sf, "r") as h5:
                arr = h5[ss][:].astype("float32")
                stat_sum = arr if stat_sum is None else stat_sum + arr
        ens.create_dataset(ss, data=(stat_sum / num_folds).astype("float16"))

    ens.close()


def nonzero_h5(h5_file: str, stat_keys):
    """Verify the HDF5 exists and has nonzero variance for each stat key.

    Args:
        h5_file: HDF5 file path.
        stat_keys: List of stat keys to check.
    """
    if not os.path.isfile(h5_file):
        return False
    try:
        with h5py.File(h5_file, "r") as h5:
            for sk in stat_keys:
                if sk not in h5 or h5[sk].shape[0] == 0:
                    print(f"{h5_file}: {sk} empty.")
                    return False
                if h5[sk][:].var(dtype="float64") == 0:
                    print(f"{h5_file}: {sk} zero var.")
                    return False
        return True
    except:
        print(f"{h5_file}: error", sys.exc_info()[0])
        return False


def split_scores(it_out_dir: str, posneg: str, vcf_dir: str, snp_stats: list):
    """Split merged SNP scores into tissue-specific files.

    Reads the flat 2D score arrays written by collect_scores() and writes
    tissue-specific subsets. Also writes targets_cov.txt for westminster_eqtl_gtex.py.

    Args:
        it_out_dir: Output iteration directory containing merge_{posneg}/.
        posneg: 'pos' or 'neg'.
        vcf_dir: Directory with tissue VCFs (*_{posneg}.vcf).
        snp_stats: List of stat keys stored in the merged file.
    """
    merge_h5_file = f"{it_out_dir}/merge_{posneg}/scores.h5"
    if not os.path.exists(merge_h5_file):
        raise FileNotFoundError(f"Merged HDF5 not found: {merge_h5_file}")

    with h5py.File(merge_h5_file, "r") as merge_h5:
        snp_index = {s.decode(): i for i, s in enumerate(merge_h5["snp"])}
        merge_scores = {ss: merge_h5[ss][:] for ss in snp_stats}

        # build targets file content from HDF5 metadata
        target_ids = [t.decode("utf-8") for t in merge_h5["target_ids"]]
        target_labels = [t.decode("utf-8") for t in merge_h5["target_labels"]]
        targets_df = pd.DataFrame(
            {"identifier": target_ids, "description": target_labels}
        )

        for tissue_vcf in glob.glob(f"{vcf_dir}/*_{posneg}.vcf"):
            tissue_label = os.path.basename(tissue_vcf).replace(
                f"_{posneg}.vcf", ""
            )

            snp_ids, chrs, poss, refs, alts = [], [], [], [], []
            for line in open(tissue_vcf):
                if not line.startswith("#"):
                    a = line.split()
                    chrm, pos, snp_id, ref, alt = a[:5]
                    if snp_id in snp_index:
                        snp_ids.append(snp_id)
                        chrs.append(chrm)
                        poss.append(int(pos))
                        refs.append(ref)
                        alts.append(alt)

            tissue_dir = f"{it_out_dir}/{tissue_label}_{posneg}"
            os.makedirs(tissue_dir, exist_ok=True)

            targets_df.to_csv(f"{tissue_dir}/targets_cov.txt", sep="\t")

            idxs = [snp_index[s] for s in snp_ids]
            with h5py.File(f"{tissue_dir}/scores.h5", "w") as th5:
                th5.create_dataset("snp", data=np.array(snp_ids, dtype="S"))
                th5.create_dataset("chr", data=np.array(chrs, dtype="S"))
                th5.create_dataset(
                    "pos", data=np.array(poss, dtype="uint32")
                )
                th5.create_dataset(
                    "ref_allele", data=np.array(refs, dtype="S")
                )
                th5.create_dataset(
                    "alt_allele", data=np.array(alts, dtype="S")
                )
                for ss in snp_stats:
                    th5.create_dataset(ss, data=merge_scores[ss][idxs])


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
