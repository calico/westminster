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

import slurm

"""
westminster_train_folds.py

Train baskerville model replicates on cross folds using given parameters and data.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options]'
  parser = OptionParser(usage)

  # train
  train_options = OptionGroup(parser, 'houndtrain.py options')
  train_options.add_option('-o', dest='out_dir',
      default='train_out',
      help='Training output directory [Default: %default]')
  
  parser.add_option_group(train_options)  
  
  # eval
  eval_options = OptionGroup(parser, 'hound_eval.py options')
  eval_options.add_option('--rank', dest='rank_corr',
      default=False, action='store_true',
      help='Compute Spearman rank correlation [Default: %default]')
  eval_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  eval_options.add_option('--json', dest='json',
      default=None,
      help='params json file.')
  eval_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  eval_options.add_option('--weight_file', dest='weight_file',
      default='model_best.h5',
      help='name of the model weight file to load [Default: model_best.h5]')
  eval_options.add_option('-t', dest='targets_file',
      default=None,
      help='targets file [Default: None]')

  parser.add_option('--step', dest='step',
      default=1, type='int',
      help='Spatial step for specificity/spearmanr [Default: %default]')
  parser.add_option_group(eval_options)

  # multi
  rep_options = OptionGroup(parser, 'replication options')
  rep_options.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  rep_options.add_option('-e', dest='conda_env',
      default='tf2.12',
      help='Anaconda environment [Default: %default]')
  rep_options.add_option('-f', dest='fold_subset',
      default=None, type='int',
      help='Run a subset of folds [Default:%default]')
  rep_options.add_option('--name', dest='name',
      default='fold', help='SLURM name prefix [Default: %default]')
  rep_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  rep_options.add_option('-q', dest='queue',
      default='titan_rtx',
      help='SLURM queue on which to run the jobs [Default: %default]')
  rep_options.add_option('--spec_queue', dest='spec_queue',
      default='',
      help='SLURM queue on which to run spec jobs [Default: %default]')
  rep_options.add_option('--spec_mem', dest='spec_mem',
      default=150000, type='int',
      help='memory requirement for spec jobs [Default: %default]')
  rep_options.add_option('--train_f3', dest='train_f3',
      default=False, action='store_true')
  parser.add_option_group(rep_options)

  (options, args) = parser.parse_args()

  # additional params
  num_folds = 6
  num_data = 1

  # subset folds
  if options.fold_subset is not None:
    num_folds = min(options.fold_subset, num_folds)

  if options.queue == 'standard':
    num_cpu = 8
    num_gpu = 0
    time_base = 64
  else:
    num_cpu = 2
    num_gpu = 1
    time_base = 24

  cmd_source = 'source /home/yuanh/.bashrc;'

  #######################################################
  # evaluate test set
  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (options.out_dir, fi, ci)

      for di in range(num_data):
        if num_data == 1:
          out_dir = '%s/eval' % it_dir
          model_file = '%s/train/%s' % (it_dir, options.weight_file)
        else:
          out_dir = '%s/eval%d' % (it_dir, di)
          model_file = '%s/train/model%d_best.h5' % (it_dir, di)

        if options.json is None:
          params_file = '%s/train/params.json' % it_dir
        else:
          params_file = options.json
        
        # check if done
        acc_file = '%s/acc.txt' % out_dir
        if os.path.isfile(acc_file):
          print('%s already generated.' % acc_file)
        else:
          cmd = cmd_source
          cmd += ' conda activate %s;' % options.conda_env
          cmd += ' echo $HOSTNAME;'
          cmd += ' hound_eval.py --f16'
          cmd += ' --head %d' % di
          cmd += ' -o %s' % out_dir
          if options.rc:
            cmd += ' --rc'
          if options.shifts:
            cmd += ' --shifts %s' % options.shifts
          if options.rank_corr:
            cmd += ' --rank'
            cmd += ' --step %d' % options.step
          cmd += ' %s' % params_file
          cmd += ' %s' % model_file
          
          if options.train_f3:
            cmd += ' %s/f3c0/data%d' % (options.out_dir, di)
          else:
            cmd += ' %s/data%d' % (it_dir, di)

          name = '%s-eval-f%dc%d' % (options.name, fi, ci)
          job = slurm.Job(cmd,
                          name=name,
                          out_file='%s.%%j.out'%out_dir,
                          err_file='%s.%%j.err'%out_dir,
                          sb_file='%s.sb'%out_dir,
                          queue=options.queue,
                          cpu=num_cpu, gpu=num_gpu,
                          mem=30000,
                          time='%d:00:00' % time_base)
          jobs.append(job)

  #######################################################
  # evaluate test specificity
  # this is modified to:
  # - save the .sb script.
  # - optionally use a target file (-t) to specify the tracks to normalize
  
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (options.out_dir, fi, ci)

      for di in range(num_data):
        if num_data == 1:
          out_dir = '%s/eval_spec' % it_dir
          model_file = '%s/train/%s' % (it_dir, options.weight_file)
        else:
          out_dir = '%s/eval%d_spec' % (it_dir, di)
          model_file = '%s/train/model%d_best.h5' % (it_dir, di)

        params_file = '%s/train/params.json' % it_dir
        # check if done
        acc_file = '%s/acc.txt' % out_dir
        if os.path.isfile(acc_file):
          print('%s already generated.' % acc_file)
        else:
          cmd = cmd_source
          cmd += ' conda activate %s;' % options.conda_env
          cmd += ' echo $HOSTNAME;'
          cmd += ' hound_eval_spec.py'
          cmd += ' --head %d' % di
          cmd += ' -o %s' % out_dir
          cmd += ' --step %d' % options.step
          # use target file to specify classes to compute specificity
          if options.targets_file is not None:  
            cmd += ' -t %s' % options.targets_file
          if options.rc:
            cmd += ' --rc'
          if options.shifts:
            cmd += ' --shifts %s' % options.shifts
          cmd += ' %s' % params_file
          cmd += ' %s' % model_file
          
          if options.train_f3:
            cmd += ' %s/f3c0/data%d' % (options.out_dir, di)
          else:
            cmd += ' %s/data%d' % (it_dir, di)
          
          name = '%s-spec-f%dc%d' % (options.name, fi, ci)
          job = slurm.Job(cmd,
                          name=name,
                          out_file='%s.%%j.out'%out_dir,
                          err_file='%s.%%j.err'%out_dir,
                          sb_file='%s.sb'%out_dir,
                          queue=options.spec_queue,
                          cpu=num_cpu, gpu=num_gpu,
                          mem=options.spec_mem,
                          time='%d:00:00' % (5*time_base))
          jobs.append(job)
   
    slurm.multi_run(jobs, max_proc=options.processes, verbose=True,
                  launch_sleep=10, update_sleep=60)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
