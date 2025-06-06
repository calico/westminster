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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
import seaborn as sns


def jointplot(ref_cors, exp_cors, label1, label2, alpha=1, out_pdf=None, title=None):
    """ "
    My preferred jointplot settings.

    Args:
        ref_cors ([float]): Reference metrics.
        exp_cors ([float]: Experiment metrics.
        label1 (str): Label for reference metrics.
        label2 (str): Label for experiment metrics.
        out_pdf (str): Output PDF file.
    """
    vmin = min(np.min(ref_cors), np.min(exp_cors))
    vmax = max(np.max(ref_cors), np.max(exp_cors))
    vspan = vmax - vmin
    vbuf = vspan * 0.1
    vmin -= vbuf
    vmax += vbuf

    g = sns.jointplot(x=ref_cors, y=exp_cors, space=0, joint_kws={"alpha": alpha})

    eps = 0.05
    g.ax_joint.text(
        1 - eps,
        eps,
        "Mean: %.4f" % np.mean(ref_cors),
        horizontalalignment="right",
        transform=g.ax_joint.transAxes,
    )
    g.ax_joint.text(
        eps,
        1 - eps,
        "Mean: %.4f" % np.mean(exp_cors),
        verticalalignment="top",
        transform=g.ax_joint.transAxes,
    )

    g.ax_joint.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="orange")
    g.ax_joint.set_xlabel(label1)
    g.ax_joint.set_ylabel(label2)

    plt.tight_layout(w_pad=0, h_pad=0)
    if title is not None:
        plt.suptitle(title)
    if out_pdf is not None:
        plt.savefig(out_pdf)


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
