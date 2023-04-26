# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import h5py
import os
import numpy as np


def collect_sad(out_dir: str, num_jobs: int):
  """Collect parallel SAD jobs' output into one HDF5.

  Args:
    out_dir (str): Output directory.
    num_jobs (int): Number of jobs to combine results from.
  """
  h5f_name = 'sad.h5'

  # count variants
  num_variants = 0
  for pi in range(num_jobs):
    # open job
    job_h5_file = '%s/job%d/%s' % (out_dir, pi, h5f_name)
    job_h5_open = h5py.File(job_h5_file, 'r')
    num_variants += len(job_h5_open['snp'])
    job_h5_open.close()

  # initialize final h5
  final_h5_file = '%s/%s' % (out_dir, h5f_name)
  final_h5_open = h5py.File(final_h5_file, 'w')

  # keep dict for string values
  final_strings = {}

  job0_h5_file = '%s/job0/%s' % (out_dir, h5f_name)
  job0_h5_open = h5py.File(job0_h5_file, 'r')
  for key in job0_h5_open.keys():
    if key in ['percentiles', 'target_ids', 'target_labels']:
      # copy
      final_h5_open.create_dataset(key, data=job0_h5_open[key])

    elif key[-4:] == '_pct':
      values = np.zeros(job0_h5_open[key].shape)
      final_h5_open.create_dataset(key, data=values)

    elif job0_h5_open[key].dtype.char == 'S':
        final_strings[key] = []

    elif job0_h5_open[key].ndim == 1:
      final_h5_open.create_dataset(key, shape=(num_variants,), dtype=job0_h5_open[key].dtype)

    else:
      num_targets = job0_h5_open[key].shape[1]
      final_h5_open.create_dataset(key, shape=(num_variants, num_targets), dtype=job0_h5_open[key].dtype)

  job0_h5_open.close()

  # set values
  vi = 0
  for pi in range(num_jobs):
    # open job
    job_h5_file = '%s/job%d/%s' % (out_dir, pi, h5f_name)
    job_h5_open = h5py.File(job_h5_file, 'r')

    # append to final
    for key in job_h5_open.keys():
      if key in ['percentiles', 'target_ids', 'target_labels']:
        # once is enough
        pass

      elif key[-4:] == '_pct':
        # average
        u_k1 = np.array(final_h5_open[key])
        x_k = np.array(job_h5_open[key])
        final_h5_open[key][:] = u_k1 + (x_k - u_k1) / (pi+1)

      else:
        if job_h5_open[key].dtype.char == 'S':
          final_strings[key] += list(job_h5_open[key])
        else:
          job_variants = job_h5_open[key].shape[0]
          try:
            final_h5_open[key][vi:vi+job_variants] = job_h5_open[key]
          except TypeError as e:
            print(e)
            print(f'{job_h5_file} ${key} has the wrong shape. Remove this file and rerun')
            exit()

    vi += job_variants
    job_h5_open.close()

  # create final string datasets
  for key in final_strings:
    final_h5_open.create_dataset(key,
      data=np.array(final_strings[key], dtype='S'))

  final_h5_open.close()


def nonzero_h5(h5_file: str, stat_keys):
  """Verify the HDF5 exists, and there are nonzero values
    for each stat key given.

  Args:
    h5_file (str): HDF5 file name.
    stat_keys ([str]): List of SNP stat keys.
  """
  if os.path.isfile(h5_file):
    try:
      with h5py.File(h5_file, 'r') as h5_open:
        for sk in stat_keys:
          sad = h5_open[sk][:]
          if (sad != 0).sum() == 0:
            return False  
        return True
    except:
      return False
  else:
    return False