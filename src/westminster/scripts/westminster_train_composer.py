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
import glob
import json
import os
import pdb
import shutil

from natsort import natsorted


"""
westminster_train_folds.py

Train baskerville model replicates on cross folds using given parameters and data.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <data1_dir> ..."
    parser = OptionParser(usage)

    # train
    train_options = OptionGroup(parser, "houndtrain.py options")
    train_options.add_option(
        "-k",
        dest="keras_fit",
        default=False,
        action="store_true",
        help="Train with Keras fit method [Default: %default]",
    )
    train_options.add_option(
        "-m",
        dest="mixed_precision",
        default=False,
        action="store_true",
        help="Train with mixed precision [Default: %default]",
    )
    train_options.add_option(
        "-o",
        dest="out_dir",
        default="train_out",
        help="Training output directory [Default: %default]",
    )
    train_options.add_option(
        "-l",
        dest="log_dir",
        default="log_out",
        help="Tensorboard log directory [Default: %default]",
    )
    train_options.add_option(
        "--restore",
        dest="restore",
        help="Restore model and continue training, from existing fold train dir [Default: %default]",
    )
    train_options.add_option(
        "--trunk",
        dest="trunk",
        default=False,
        action="store_true",
        help="Restore only model trunk [Default: %default]",
    )
    train_options.add_option(
        "--tfr_train",
        dest="tfr_train_pattern",
        default=None,
        help="Training TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]",
    )
    train_options.add_option(
        "--tfr_eval",
        dest="tfr_eval_pattern",
        default=None,
        help="Evaluation TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]",
    )
    parser.add_option_group(train_options)

    # eval
    eval_options = OptionGroup(parser, "hound_eval.py options")
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
        "--checkpoint",
        dest="checkpoint",
        default=False,
        action="store_true",
        help="Restart training from checkpoint [Default: %default]",
    )
    rep_options.add_option(
        "-f",
        dest="fold_subset",
        default=None,
        type="int",
        help="Run a subset of folds [Default:%default]",
    )
    rep_options.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    rep_options.add_option(
        "-r", "--restart", dest="restart", default=False, action="store_true"
    )
    rep_options.add_option(
        "--setup",
        dest="setup",
        default=False,
        action="store_true",
        help="Setup folds data directory only [Default: %default]",
    )
    rep_options.add_option(
        "--spec_off", dest="spec_off", default=False, action="store_true"
    )
    rep_options.add_option(
        "--eval_off", dest="eval_off", default=False, action="store_true"
    )
    rep_options.add_option(
        "--eval_train_off", dest="eval_train_off", default=False, action="store_true"
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

    if not options.restart and os.path.isdir(options.out_dir):
        print("Output directory %s exists. Please remove." % options.out_dir)
        exit(1)
    os.makedirs(options.out_dir, exist_ok=True)
    os.makedirs(options.log_dir, exist_ok=True)

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_train = params["train"]

    # copy params into output directory
    shutil.copy(params_file, "%s/params.json" % options.out_dir)

    # read data parameters
    num_data = len(data_dirs)
    data_stats_file = "%s/statistics.json" % data_dirs[0]
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # count folds
    num_folds = len([dkey for dkey in data_stats if dkey.startswith("fold")])

    # subset folds
    if options.fold_subset is not None:
        num_folds = min(options.fold_subset, num_folds)

    # arrange data
    for ci in range(options.crosses):
        for fi in range(num_folds):
            rep_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)
            os.makedirs(rep_dir, exist_ok=True)

            # make data directories
            for di in range(num_data):
                rep_data_dir = "%s/data%d" % (rep_dir, di)
                if not os.path.isdir(rep_data_dir):
                    make_rep_data(data_dirs[di], rep_data_dir, fi, ci)

    if options.setup:
        exit(0)

    #######################################################
    # train

    train_jobs = []

    for ci in range(options.crosses):
        for fi in range(num_folds):
            rep_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)
            rep_log_dir = "%s/f%dc%d" % (options.log_dir, fi, ci)

            train_dir = "%s/train" % rep_dir
            if options.restart and not options.checkpoint and os.path.isdir(train_dir):
                print("%s found and skipped." % rep_dir)

            else:
                # collect data directories
                rep_data_dirs = []
                for di in range(num_data):
                    rep_data_dirs.append("%s/data%d" % (rep_dir, di))

                # train command
                cmd = "python3 -m baskerville.scripts.hound_train"
                cmd += " %s" % options_string(options, train_options, rep_dir, rep_log_dir)
                cmd += " %s %s" % (params_file, " ".join(rep_data_dirs))
                train_jobs.append(cmd)

    # write to the outdir the train_jobs
    with open(options.out_dir + "/train_jobs.txt", "w") as f:
        for item in train_jobs:
            f.write("%s\n" % item)

    #######################################################
    # evaluate training set

    eval_jobs = []

    if not options.eval_train_off:
        for ci in range(options.crosses):
            for fi in range(num_folds):
                it_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)

                for di in range(num_data):
                    if num_data == 1:
                        out_dir = "%s/eval_train" % it_dir
                        model_file = "%s/train/model_check.h5" % it_dir
                    else:
                        out_dir = "%s/eval%d_train" % (it_dir, di)
                        model_file = "%s/train/model%d_check.h5" % (it_dir, di)

                    # check if done
                    acc_file = "%s/acc.txt" % out_dir
                    if os.path.isfile(acc_file):
                        print("%s already generated." % acc_file)
                    else:
                        # hound evaluate
                        cmd = "python3 -m baskerville.scripts.hound_eval"
                        cmd += " --head %d" % di
                        cmd += " -o %s" % out_dir
                        if options.rc:
                            cmd += " --rc"
                        if options.shifts:
                            cmd += " --shifts %s" % options.shifts
                        cmd += " --split train"
                        cmd += " %s" % params_file
                        cmd += " %s" % model_file
                        cmd += " %s/data%d" % (it_dir, di)
                        eval_jobs.append(cmd)

    #######################################################
    # evaluate test set

    if not options.eval_off:
        for ci in range(options.crosses):
            for fi in range(num_folds):
                it_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)

                for di in range(num_data):
                    if num_data == 1:
                        out_dir = "%s/eval" % it_dir
                        model_file = "%s/train/model_best.h5" % it_dir
                    else:
                        out_dir = "%s/eval%d" % (it_dir, di)
                        model_file = "%s/train/model%d_best.h5" % (it_dir, di)

                    # check if done
                    acc_file = "%s/acc.txt" % out_dir
                    if os.path.isfile(acc_file):
                        print("%s already generated." % acc_file)
                    else:
                        cmd = "python3 -m baskerville.scripts.hound_eval"
                        cmd += " --head %d" % di
                        cmd += " -o %s" % out_dir
                        if options.rc:
                            cmd += " --rc"
                        if options.shifts:
                            cmd += " --shifts %s" % options.shifts
                        if options.rank_corr:
                            cmd += " --rank"
                            cmd += " --step %d" % options.step
                        cmd += " %s" % params_file
                        cmd += " %s" % model_file
                        cmd += " %s/data%d" % (it_dir, di)
                        eval_jobs.append(cmd)

    #######################################################
    # evaluate test specificity

    if not options.spec_off:
        for ci in range(options.crosses):
            for fi in range(num_folds):
                it_dir = "%s/f%dc%d" % (options.out_dir, fi, ci)

                for di in range(num_data):
                    if num_data == 1:
                        out_dir = "%s/eval_spec" % it_dir
                        model_file = "%s/train/model_best.h5" % it_dir
                    else:
                        out_dir = "%s/eval%d_spec" % (it_dir, di)
                        model_file = "%s/train/model%d_best.h5" % (it_dir, di)

                    # check if done
                    acc_file = "%s/acc.txt" % out_dir
                    if os.path.isfile(acc_file):
                        print("%s already generated." % acc_file)
                    else:
                        cmd = "python3 -m baskerville.scripts.hound_eval_spec"
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
                        eval_jobs.append(cmd)

    with open(options.out_dir + "/eval_jobs.txt", "w") as f:
        for item in eval_jobs:
            f.write("%s\n" % item)


def make_rep_data(data_dir, rep_data_dir, fi, ci):
    # read data parameters
    data_stats_file = "%s/statistics.json" % data_dir
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # sequences per fold
    fold_seqs = []
    dfi = 0
    while "fold%d_seqs" % dfi in data_stats:
        fold_seqs.append(data_stats["fold%d_seqs" % dfi])
        del data_stats["fold%d_seqs" % dfi]
        dfi += 1
    num_folds = dfi

    # split folds into train/valid/test
    test_fold = fi
    valid_fold = (fi + 1 + ci) % num_folds
    train_folds = [
        fold for fold in range(num_folds) if fold not in [valid_fold, test_fold]
    ]

    # clear existing directory
    if os.path.isdir(rep_data_dir):
        shutil.rmtree(rep_data_dir)

    # make data directory
    os.makedirs(rep_data_dir, exist_ok=True)

    # dump data stats
    data_stats["test_seqs"] = fold_seqs[test_fold]
    data_stats["valid_seqs"] = fold_seqs[valid_fold]
    data_stats["train_seqs"] = sum([fold_seqs[tf] for tf in train_folds])
    with open("%s/statistics.json" % rep_data_dir, "w") as data_stats_open:
        json.dump(data_stats, data_stats_open, indent=4)

    # set sequence tvt
    try:
        seqs_bed_out = open("%s/sequences.bed" % rep_data_dir, "w")
        for line in open("%s/sequences.bed" % data_dir):
            a = line.split()
            sfi = int(a[-1].replace("fold", ""))
            if sfi == test_fold:
                a[-1] = "test"
            elif sfi == valid_fold:
                a[-1] = "valid"
            else:
                a[-1] = "train"
            print("\t".join(a), file=seqs_bed_out)
        seqs_bed_out.close()
    except (ValueError, FileNotFoundError):
        pass

    # copy targets
    shutil.copy("%s/targets.txt" % data_dir, "%s/targets.txt" % rep_data_dir)

    # sym link tfrecords
    rep_tfr_dir = "%s/tfrecords" % rep_data_dir
    os.mkdir(rep_tfr_dir)

    # test tfrecords
    ti = 0
    test_tfrs = natsorted(
        glob.glob("%s/tfrecords/fold%d-*.tfr" % (data_dir, test_fold))
    )
    for test_tfr in test_tfrs:
        test_tfr = os.path.abspath(test_tfr)
        test_rep_tfr = "%s/test-%d.tfr" % (rep_tfr_dir, ti)
        os.symlink(test_tfr, test_rep_tfr)
        ti += 1

    # valid tfrecords
    ti = 0
    valid_tfrs = natsorted(
        glob.glob("%s/tfrecords/fold%d-*.tfr" % (data_dir, valid_fold))
    )
    for valid_tfr in valid_tfrs:
        valid_tfr = os.path.abspath(valid_tfr)
        valid_rep_tfr = "%s/valid-%d.tfr" % (rep_tfr_dir, ti)
        os.symlink(valid_tfr, valid_rep_tfr)
        ti += 1

    # train tfrecords
    ti = 0
    train_tfrs = []
    for tfi in train_folds:
        train_tfrs += natsorted(
            glob.glob("%s/tfrecords/fold%d-*.tfr" % (data_dir, tfi))
        )
    for train_tfr in train_tfrs:
        train_tfr = os.path.abspath(train_tfr)
        train_rep_tfr = "%s/train-%d.tfr" % (rep_tfr_dir, ti)
        os.symlink(train_tfr, train_rep_tfr)
        ti += 1


def options_string(options, train_options, rep_dir, rep_log_dir):
    options_str = ""

    for opt in train_options.option_list:
        opt_str = opt.get_opt_string()
        opt_value = options.__dict__[opt.dest]

        # wrap askeriks in ""
        if type(opt_value) == str and opt_value.find("*") != -1:
            opt_value = '"%s"' % opt_value

        # no value for bools
        elif type(opt_value) == bool:
            if not opt_value:
                opt_str = ""
            opt_value = ""

        # skip Nones
        elif opt_value is None:
            opt_str = ""
            opt_value = ""

        # modify
        elif opt.dest == "out_dir":
            opt_value = "%s/train" % rep_dir

        # modify
        elif opt.dest == "log_dir":
            opt_value = "%s/log" % rep_log_dir

        # find matching restore
        elif opt.dest == "restore":
            fold_dir_mid = rep_dir.split("/")[-1]
            if options.trunk:
                opt_value = "%s/%s/train/model_trunk.h5" % (opt_value, fold_dir_mid)
            else:
                opt_value = "%s/%s/train/model_best.h5" % (opt_value, fold_dir_mid)

        options_str += " %s %s" % (opt_str, opt_value)

    return options_str


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
