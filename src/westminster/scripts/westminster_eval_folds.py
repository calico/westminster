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

from optparse import OptionParser, OptionGroup
import json
import os
import pdb

import slurm

"""
westminster_eval_folds.py

Evaluate baskerville model replicates on cross folds using given parameters and data.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <data1_dir> ..."
    parser = OptionParser(usage)

    # eval
    eval_options = OptionGroup(parser, "hound_eval.py options")
    eval_options.add_option(
        "-o",
        dest="out_dir",
        default="models",
        help="Training output directory [Default: %default]",
    )
    eval_options.add_option(
        "--rank",
        dest="rank_corr",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %default]",
    )
    eval_options.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    eval_options.add_option(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %default]",
    )
    eval_options.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--step",
        dest="step",
        default=1,
        type="int",
        help="Spatial step for specificity/spearmanr [Default: %default]",
    )
    parser.add_option_group(eval_options)

    # multi
    rep_options = OptionGroup(parser, "replication options")
    rep_options.add_option(
        "-c",
        dest="crosses",
        default=1,
        type="int",
        help="Number of cross-fold rounds [Default:%default]",
    )
    rep_options.add_option(
        "-e",
        dest="conda_env",
        default="tf15b",
        help="Anaconda environment [Default: %default]",
    )
    rep_options.add_option(
        "-f",
        dest="fold_subset",
        default=None,
        type="int",
        help="Run a subset of folds [Default:%default]",
    )
    rep_options.add_option(
        "--f_list",
        dest="fold_subset_list",
        default=None,
        help="Run a subset of folds (encoded as comma-separated string) [Default:%default]",
    )
    rep_options.add_option(
        "--name",
        dest="name",
        default="fold",
        help="SLURM name prefix [Default: %default]",
    )
    rep_options.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    rep_options.add_option(
        "-q",
        dest="queue",
        default="titan_rtx",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    rep_options.add_option(
        "-r", "--restart", dest="restart", default=False, action="store_true"
    )
    rep_options.add_option(
        "--spec",
        dest="spec",
        default=False,
        action="store_true",
        help="Specificity evaluation [Default: %default]",
    )
    parser.add_option_group(rep_options)

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error("Must provide parameters and data directory.")
    else:
        params_file = os.path.abspath(args[0])
        data_dirs = [os.path.abspath(arg) for arg in args[1:]]

    #######################################################
    # prep work

    # read data parameters
    num_data = len(data_dirs)
    data_stats_file = f"{data_dirs[0]}/statistics.json"
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
        time_base = 64
    else:
        num_cpu = 4
        num_gpu = 1
        time_base = 24

    #######################################################
    # evaluate folds

    jobs = []

    for ci in range(options.crosses):
        for fi in fold_index:
            it_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)

            for di in range(num_data):
                model_file = get_model_file(it_dir, di)
                if os.path.isfile(model_file):
                    if num_data == 1:
                        eval_dir = f"{it_dir}/eval"
                    else:
                        eval_dir = f"{it_dir}/eval{di}"
                    os.makedirs(eval_dir, exist_ok=True)

                    for ei in range(num_folds):
                        eval_fold_dir = f"{eval_dir}/fold{ei}"

                        # symlink test metrics
                        if fi == ei and not os.path.isfile(f"{eval_dir}/test.out"):
                            os.symlink(f"fold{ei}", f"{eval_dir}/test")
                            os.symlink(f"fold{ei}.out", f"{eval_dir}/test.out")

                        # check if done
                        acc_file = f"{eval_fold_dir}/acc.txt"
                        if os.path.isfile(acc_file):
                            print(f"{acc_file} already generated.")
                        else:
                            # hound evaluate
                            cmd = (
                                (". %s; " % os.environ["BASKERVILLE_CONDA"])
                                if "BASKERVILLE_CONDA" in os.environ
                                else ""
                            )
                            cmd += "conda activate %s;" % options.conda_env
                            cmd += " echo $HOSTNAME;"
                            cmd += " hound_eval.py"
                            cmd += f" --head {di}"
                            cmd += f" -o {eval_fold_dir}"
                            if options.rc:
                                cmd += " --rc"
                            if options.save:
                                cmd += " --save"
                            if options.shifts:
                                cmd += f" --shifts {options.shifts}"
                            cmd += f" --split fold{ei}"
                            cmd += f" {params_file}"
                            cmd += f" {model_file}"
                            cmd += f" {data_dirs[di]}"

                            name = f"{options.name}-eval-f{fi}e{ei}"
                            job = slurm.Job(
                                cmd,
                                name=name,
                                out_file=f"{eval_fold_dir}.out",
                                err_file=f"{eval_fold_dir}.err",
                                queue=options.queue,
                                cpu=num_cpu,
                                gpu=num_gpu,
                                mem=30000,
                                time="%d:00:00" % time_base,
                            )
                            jobs.append(job)

    #######################################################
    # evaluate test specificity

    if options.spec:
        for ci in range(options.crosses):
            for fi in fold_index:
                it_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)

                for di in range(num_data):
                    model_file = get_model_file(it_dir, di)
                    if os.path.isfile(model_file):
                        if num_data == 1:
                            out_dir = f"{it_dir}/spec"
                        else:
                            out_dir = f"{it_dir}/spec{di}"

                        # check if done
                        acc_file = "%s/acc.txt" % out_dir
                        if os.path.isfile(acc_file):
                            print(f"{acc_file} already generated.")
                        else:
                            cmd = (
                                (". %s; " % os.environ["BASKERVILLE_CONDA"])
                                if "BASKERVILLE_CONDA" in os.environ
                                else ""
                            )
                            cmd += "conda activate %s;" % options.conda_env
                            cmd += " echo $HOSTNAME;"
                            cmd += " hound_eval_spec.py"
                            cmd += " --head %d" % di
                            cmd += " -o %s" % out_dir
                            cmd += " --step %d" % options.step
                            if options.rc:
                                cmd += " --rc"
                            if options.shifts:
                                cmd += " --shifts %s" % options.shifts
                            cmd += " %s" % params_file
                            cmd += " %s" % model_file
                            cmd += " %s/data%d" % (it_dir, di)

                            name = "%s-spec-f%dc%d" % (options.name, fi, ci)
                            job = slurm.Job(
                                cmd,
                                name=name,
                                out_file=f"{out_dir}.out",
                                err_file=f"{out_dir}.err",
                                queue=options.queue,
                                cpu=num_cpu,
                                gpu=num_gpu,
                                mem=90000,
                                time="%d:00:00" % (3 * time_base),
                            )
                            jobs.append(job)

    slurm.multi_run(
        jobs, max_proc=options.processes, verbose=True, launch_sleep=10, update_sleep=60
    )


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
