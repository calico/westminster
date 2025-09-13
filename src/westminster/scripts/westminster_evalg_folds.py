#!/usr/bin/env python
# Copyright 2019 Calico LLC

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
from optparse import OptionParser
import json
import os

import slurm

"""
westminster_evalg_folds.py

Measure accuracy at gene-level for multiple model replicates.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <data1_dir> ..."
    parser = OptionParser(usage)

    # eval
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate [Default: %(default)s]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="models",
        help="Training output directory [Default: %default]",
    )
    parser.add_option(
        "--pseudo_qtl",
        dest="pseudo_qtl",
        default=None,
        type="float",
        help="Quantile of coverage to add as pseudo counts to genes [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %default]",
    )
    parser.add_option(
        "--save_span",
        dest="save_span",
        default=False,
        action="store_true",
        help="Store predicted/measured gene span coverage profiles [Default: %default]",
    )
    parser.add_option(
        "--seq_step",
        dest="seq_step",
        default=1,
        type="int",
        help="Compute only every seq_step sequence [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )

    # folds
    parser.add_option(
        "-c",
        dest="crosses",
        default=1,
        type="int",
        help="Number of cross-fold rounds [Default:%default]",
    )
    parser.add_option(
        "-d",
        dest="dataset_i",
        default=None,
        type="int",
        help="Dataset index [Default:%default]",
    )
    parser.add_option(
        "-e",
        dest="conda_env",
        default="tf15b",
        help="Anaconda environment [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="fold_subset",
        default=None,
        type="int",
        help="Run a subset of folds [Default:%default]",
    )
    parser.add_option(
        "--f_list",
        dest="fold_subset_list",
        default=None,
        help="Run a subset of folds (encoded as comma-separated string) [Default:%default]",
    )
    parser.add_option(
        "-g",
        dest="genes_gtf",
        default="%s/genes/gencode41/gencode41_basic_protein.gtf"
        % os.environ.get("BORZOI_HG38", "hg38"),
    )
    parser.add_option(
        "--name",
        dest="name",
        default="test",
        help="SLURM name prefix [Default: %default]",
    )
    parser.add_option("-q", dest="queue", default="geforce")
    parser.add_option(
        "-s",
        dest="sub_dir",
        default="evalg",
        help="Output subdirectory within the fold directories [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error("Must provide parameters file and data directory")
    else:
        params_file = args[0]
        data_dir = os.path.abspath(args[1])

    #######################################################
    # prep work

    # read data parameters
    data_stats_file = f"{data_dir}/statistics.json"
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # count folds
    num_folds = len([dkey for dkey in data_stats if dkey.startswith("fold")])

    # select specific folds to evaluate
    if options.fold_subset_list is None:
        # focus on initial folds
        if options.fold_subset is None:
            num_folds_score = num_folds
        else:
            num_folds_score = min(options.fold_subset, num_folds)
        fold_index = [fold_i for fold_i in range(num_folds_score)]
    else:
        fold_index = [int(fold_str) for fold_str in options.fold_subset_list.split(",")]

    if options.queue == "standard":
        num_cpu = 16
        num_gpu = 0
    else:
        num_cpu = 8
        num_gpu = 1

    ################################################################
    # evaluate folds

    jobs = []

    for ci in range(options.crosses):
        for fi in fold_index:
            it_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)

            model_file = get_model_file(it_dir, options.head_i)
            eval_dir = "%s/%s-%d" % (it_dir, options.sub_dir, options.head_i)

            # check if done
            acc_file = f"{eval_dir}/acc.txt"
            if os.path.isfile(acc_file):
                print(f"{acc_file} already generated.")
            else:
                # evaluate
                cmd = (
                    (". %s; " % os.environ["BASKERVILLE_CONDA"])
                    if "BASKERVILLE_CONDA" in os.environ
                    else ""
                )
                cmd += f" echo $HOSTNAME;"
                cmd += f" conda activate {options.conda_env};"
                cmd += f" hound_eval_genes.py"
                cmd += f" --head {options.head_i}"
                cmd += f" -o {eval_dir}"
                cmd += f" --seq_step {options.seq_step}"
                if options.pseudo_qtl is not None:
                    cmd += f" --pseudo_qtl {options.pseudo_qtl:.2f}"
                if options.rc:
                    cmd += " --rc"
                if options.save_span:
                    cmd += " --save_span"
                if options.shifts:
                    cmd += f" --shifts {options.shifts}"
                if options.span:
                    cmd += " --span"
                    job_mem = 240000
                else:
                    job_mem = 150000
                if options.targets_file is not None:
                    cmd += f" -t {options.targets_file}"
                cmd += f" {params_file}"
                cmd += f" {model_file}"
                cmd += f" {it_dir}/data{options.head_i}"
                cmd += f" {options.genes_gtf}"

                name = "%s-evalg-f%dc%d" % (options.name, fi, ci)
                j = slurm.Job(
                    cmd,
                    name=name,
                    out_file=f"{eval_dir}.out",
                    err_file=f"{eval_dir}.err",
                    queue=options.queue,
                    cpu=num_cpu,
                    gpu=num_gpu,
                    mem=job_mem,
                    time="4-00:00:00",
                )
                jobs.append(j)

    slurm.multi_run(jobs, verbose=True)


def get_model_file(it_dir, di):
    """Find model file in it_dir, robust to pytorch or tensorflow."""
    # pytorch
    model_file = f"{it_dir}/train/model_best.pth"
    if not os.path.isfile(model_file):
        # single data tensorflow
        model_file = f"{it_dir}/train/model_best.h5"
        if not os.path.isfile(model_file):
            # multi data tensorflow
            model_file = f"{it_dir}/train/model{di}_best.h5"
            if not os.path.isfile(model_file):
                model_file = ""
                print(f"No model in {it_dir}")
    return model_file


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
