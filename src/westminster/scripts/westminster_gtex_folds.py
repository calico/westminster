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
import os
import shutil

import h5py
import numpy as np

import slurm
import util

# from westminster.multi import collect_scores, nonzero_h5
from baskerville_torch.scripts.hound_snp_folds import snp_folds

"""
westminster_gtex_folds.py

Benchmark Baskerville model replicates on GTEx eQTL classification task.
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
        "-g",
        dest="genes_gtf",
        default=None,
        help="Trigger gene scoring mode; provide GTF for gene definitions (skips splitting & classifiers)",
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
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="In gene scoring mode, aggregate entire gene span",
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
        required=True,
        help="File specifying target indexes and labels in table format",
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
        "--local", dest="local", default=False, action="store_true", help="Run locally"
    )
    fold_group.add_argument(
        "--name", dest="name", default="snp", help="SLURM name prefix"
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
    # GTEx directory
    gtex_group.add_argument(
        "--gtex",
        dest="gtex_vcf_dir",
        default="/home/drk/seqnn/data/gtex_fine/susie_pip90r",
        help="Directory with GTEx VCF files",
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

    # subset folds
    fold_index = [fold_i for fold_i in range(args.num_folds)]

    # subset folds (list)
    if args.fold_subset_list is not None:
        fold_index = [int(fold_str) for fold_str in args.fold_subset_list.split(",")]

    # extract output subdirectory name
    gtex_out_dir = args.out_dir

    # split SNP stats
    snp_stats = args.snp_stats.split(",")

    ################################################################
    # score SNPs

    # merge study/tissue variants
    mpos_vcf_file = f"{args.gtex_vcf_dir}/pos_merge.vcf"
    mneg_vcf_file = f"{args.gtex_vcf_dir}/neg_merge.vcf"

    # embed output in the models directory
    args.embed = True

    # score negative SNPs
    args.vcf_file = mneg_vcf_file
    args.out_dir = f"{gtex_out_dir}/merge_neg"
    snp_folds(args)

    # score positive SNPs
    args.vcf_file = mpos_vcf_file
    args.out_dir = f"{gtex_out_dir}/merge_pos"
    snp_folds(args)

    ################################################################
    # split study/tissue variants

    for ci in range(args.crosses):
        for fi in fold_index:
            it_out_dir = f"{args.models_dir}/f{fi}c{ci}/{gtex_out_dir}"
            print(it_out_dir)

            # split positives
            split_scores(it_out_dir, "pos", args.gtex_vcf_dir, snp_stats)

            # split negatives
            split_scores(it_out_dir, "neg", args.gtex_vcf_dir, snp_stats)

    # split ensemble positives
    ens_out_dir = f"{args.models_dir}/ensemble/{gtex_out_dir}"
    split_scores(ens_out_dir, "pos", args.gtex_vcf_dir, snp_stats)

    # split ensemble negatives
    split_scores(ens_out_dir, "neg", args.gtex_vcf_dir, snp_stats)

    ################################################################
    # fit classifiers

    if args.genes_gtf is None:

        # SNPs (random forest)
        # cmd_base = "westminster_classify.py -f 8 -i 20 -n 512 -s"
        # SNPs (xgboost)
        cmd_base = "westminster_classify.py -f 8 -i 20 -n 96 -s -x"
        # indels
        # cmd_base = 'westminster_classify.py -f 6 -i 64 -s'
        cmd_base += f" --msl {args.msl}"

        if args.class_targets_file is not None:
            cmd_base += f" -t {args.class_targets_file}"

        jobs = []
        for ci in range(args.crosses):
            for fi in fold_index:
                it_dir = f"{args.models_dir}/f{fi}c{ci}"
                it_out_dir = f"{it_dir}/{gtex_out_dir}"

                for gtex_pos_vcf in glob.glob(f"{args.gtex_vcf_dir}/*_pos.vcf"):
                    tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
                    sad_pos = f"{it_out_dir}/{tissue}_pos/scores.h5"
                    sad_neg = f"{it_out_dir}/{tissue}_neg/scores.h5"
                    for snp_stat in snp_stats:
                        class_out_dir = f"{it_out_dir}/{tissue}_class-{snp_stat}"
                        if args.class_name is not None:
                            class_out_dir += f"-{args.class_name}"
                        if not os.path.isfile(f"{class_out_dir}/stats.txt"):
                            cmd_class = (
                                f"{cmd_base} -o {class_out_dir} --stat {snp_stat}"
                            )
                            cmd_class += f" {sad_pos} {sad_neg}"
                            if args.local:
                                jobs.append(cmd_class)
                            else:
                                j = slurm.Job(
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
                class_out_dir = f"{ens_out_dir}/{tissue}_class-{snp_stat}"
                if args.class_name is not None:
                    class_out_dir += f"-{args.class_name}"
                if not os.path.isfile(f"{class_out_dir}/stats.txt"):
                    cmd_class = f"{cmd_base} -o {class_out_dir} --stat {snp_stat}"
                    cmd_class += f" {sad_pos} {sad_neg}"
                    if args.local:
                        jobs.append(cmd_class)
                    else:
                        j = slurm.Job(
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
            util.exec_par(jobs, 3, verbose=True)
        else:
            slurm.multi_run(jobs, verbose=True)

    ################################################################
    # coefficient analysis

    if args.genes_gtf is None:
        cmd_base = f"westminster_gtex_coef.py -g {args.gtex_vcf_dir}"
    else:
        cmd_base = f"westminster_gtexg_coef.py -g {args.gtex_vcf_dir}"

    jobs = []
    for ci in range(args.crosses):
        for fi in fold_index:
            it_dir = f"{args.models_dir}/f{fi}c{ci}"
            it_out_dir = f"{it_dir}/{gtex_out_dir}"
            for snp_stat in snp_stats:
                coef_out_dir = f"{it_out_dir}/coef-{snp_stat}"

                cmd_coef = cmd_base
                cmd_coef += f" -o {coef_out_dir}"
                cmd_coef += f" -s {snp_stat}"
                cmd_coef += f" {it_out_dir}"
                if args.local:
                    jobs.append(cmd_coef)
                else:
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
    for snp_stat in snp_stats:
        coef_out_dir = f"{ens_out_dir}/coef-{snp_stat}"

        cmd_coef = cmd_base
        cmd_coef += f" -o {coef_out_dir}"
        cmd_coef += f" -s {snp_stat}"
        cmd_coef += f" {ens_out_dir}"

        if args.local:
            jobs.append(cmd_coef)
        else:
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

    if args.local:
        util.exec_par(jobs, 3, verbose=True)
    else:
        slurm.multi_run(jobs, verbose=True)


def split_scores(it_out_dir: str, posneg: str, vcf_dir: str, snp_stats):
    """Split merged SNP scores into tissue-specific files.

    Supports both standard SNP-level scoring and gene-specific scoring.
    Gene-specific merged files contain the extra datasets:
      gene_ids, snp_idx, gene_idx
    where each row in the stats corresponds to a (snp_idx, gene_idx) pair.

    Args:
        it_out_dir (str): Output iteration directory containing merge_{posneg}.
        posneg (str): 'pos' or 'neg'.
        vcf_dir (str): Directory with tissue VCFs (*_{posneg}.vcf).
        snp_stats (list[str]): List of statistics stored in merged file.
    """
    merge_dir = f"{it_out_dir}/merge_{posneg}"
    targets_file = f"{merge_dir}/targets.txt"
    merge_h5_file = f"{merge_dir}/scores.h5"

    if not os.path.exists(merge_h5_file):
        raise FileNotFoundError(f"Merged HDF5 file not found: {merge_h5_file}")

    with h5py.File(merge_h5_file, "r") as merge_h5:
        # detect gene-specific format
        gene_specific = (
            "gene_ids" in merge_h5 and "snp_idx" in merge_h5 and "gene_idx" in merge_h5
        )

        merge_scores = {ss: merge_h5[ss][:] for ss in snp_stats}
        snps = [s.decode("utf-8") for s in merge_h5["snp"]]
        snp_index = {snp_id: i for i, snp_id in enumerate(snps)}

        if gene_specific:
            gene_ids = [g.decode("utf-8") for g in merge_h5["gene_ids"]]
            snp_idx_map = merge_h5["snp_idx"][:]  # row -> base snp index
            gene_idx_map = merge_h5["gene_idx"][:]  # row -> base gene index
            snp_to_rows = {}
            for row_i, base_snp_i in enumerate(snp_idx_map):
                snp_id = snps[int(base_snp_i)]
                snp_to_rows.setdefault(snp_id, []).append(row_i)
        else:
            snp_to_rows = None

        # iterate VCFs
        for tissue_vcf in glob.glob(f"{vcf_dir}/*_{posneg}.vcf"):
            tissue_label = os.path.basename(tissue_vcf)
            tissue_label = tissue_label.replace("_pos.vcf", "").replace("_neg.vcf", "")

            tissue_snps = []
            tissue_chr = []
            tissue_pos = []
            tissue_ref = []
            tissue_alt = []
            # gene-specific row accumulation
            selected_rows = []

            with open(tissue_vcf) as vf:
                for line in vf:
                    if line.startswith("#"):
                        continue
                    a = line.split()
                    chrm, pos, snp_id, ref, alt = a[:5]
                    if snp_id not in snp_index:
                        raise ValueError(
                            f"SNP {snp_id} from {tissue_vcf} not found in {merge_h5_file}"
                        )
                    tissue_snps.append(snp_id)
                    tissue_chr.append(chrm)
                    tissue_pos.append(int(pos))
                    tissue_ref.append(ref)
                    tissue_alt.append(alt)
                    if gene_specific:
                        selected_rows.extend(snp_to_rows.get(snp_id, []))

            tissue_dir = f"{it_out_dir}/{tissue_label}_{posneg}"
            os.makedirs(tissue_dir, exist_ok=True)
            if os.path.exists(targets_file):
                shutil.copyfile(targets_file, f"{tissue_dir}/targets.txt")

            out_h5_path = f"{tissue_dir}/scores.h5"
            with h5py.File(out_h5_path, "w") as out_h5:
                out_h5.create_dataset("progress_status", data="completed".encode("utf-8"))
                out_h5.create_dataset("snp", data=np.array(tissue_snps, dtype="S"))
                out_h5.create_dataset("chr", data=np.array(tissue_chr, dtype="S"))
                out_h5.create_dataset("pos", data=np.array(tissue_pos, dtype="uint32"))
                out_h5.create_dataset("ref_allele", data=np.array(tissue_ref, dtype="S"))
                out_h5.create_dataset("alt_allele", data=np.array(tissue_alt, dtype="S"))

                if gene_specific:
                    # order rows grouped by tissue SNP order
                    ordered_rows = []
                    for snp_id in tissue_snps:
                        ordered_rows.extend(sorted(snp_to_rows.get(snp_id, [])))
                    ordered_rows = np.array(ordered_rows, dtype=int)

                    used_gene_indices = sorted({int(gene_idx_map[r]) for r in ordered_rows})
                    oldgene_to_new = {g: i for i, g in enumerate(used_gene_indices)}
                    new_gene_ids = [gene_ids[g] for g in used_gene_indices]
                    out_h5.create_dataset("gene_ids", data=np.array(new_gene_ids, dtype="S"))

                    oldsnp_to_new = {snp_index[snp_id]: i for i, snp_id in enumerate(tissue_snps)}
                    new_snp_idx = np.array([oldsnp_to_new[int(snp_idx_map[r])] for r in ordered_rows], dtype="int64")
                    new_gene_idx = np.array([oldgene_to_new[int(gene_idx_map[r])] for r in ordered_rows], dtype="int64")
                    out_h5.create_dataset("snp_idx", data=new_snp_idx)
                    out_h5.create_dataset("gene_idx", data=new_gene_idx)

                    for ss in snp_stats:
                        out_h5.create_dataset(ss, data=merge_scores[ss][ordered_rows].astype("float16"))
                else:
                    merged_indices = [snp_index[s] for s in tissue_snps]
                    for ss in snp_stats:
                        out_h5.create_dataset(ss, data=merge_scores[ss][merged_indices].astype("float16"))


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
