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

from scipy.stats import wilcoxon, ttest_rel


def stat_tests(ref_cors, exp_cors, alternative: str):
    """
    Compute Mann-Whitney and t-tests of reference versus experiment metrics.

    Args:
        ref_cors ([float]): Reference metrics.
        exp_cors ([float]: Experiment metrics.
        alternative (str): Alternative argument passed to statistical test.
    Returns:
        (float, float): Mann-Whitney p-value, t-test p-value.
    """
    # hack for the common situtation where I have more reference
    # crosses than experiment crosses
    if len(ref_cors) == 2 * len(exp_cors):
        ref_cors = [ref_cors[i] for i in range(len(ref_cors)) if i % 2 == 0]

    _, mwp = wilcoxon(exp_cors, ref_cors, alternative=alternative)
    tt, tp = ttest_alt(exp_cors, ref_cors, alternative=alternative)

    return mwp, tp


def ttest_alt(a, b, alternative="two-sided"):
    """Compute t-tests with alternative hypothesis."""
    tt, tp = ttest_rel(a, b)

    if alternative == "greater":
        if tt > 0:
            tp = 1 - (1 - tp) / 2
        else:
            tp /= 2
    elif alternative == "less":
        if tt <= 0:
            tp /= 2
        else:
            tp = 1 - (1 - tp) / 2

    return tt, tp
