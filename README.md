# westminster
Benchmarking baskerville models for regulatory sequence activity prediction

### Installation

`git clone git@github.com:calico/westminster.git`
`cd westminster`
`pip install .`

To set up the required environment variables:
`cd westminster`
`conda activate <conda_env>`
`./env_vars.sh`

*Note:* Change the line of code at the top of './env_vars.sh' to your local path.

Alternatively, the environment variables can be set manually:
```sh
export WESTMINSTER_DIR=/home/<user_path>/westminster
export PATH=$WESTMINSTER_DIR/src/westminster/scripts:$PATH
export PYTHONPATH=$WESTMINSTER_DIR/src/westminster/scripts:$PYTHONPATH
```

### Benchmark data

The fine-mapped GTEx QTL sets used by `westminster_{eqtl,sqtl,paqtl}_folds.py` are
built and released by [calico/qtl-bench](https://github.com/calico/qtl-bench).
Fetch a release and point `--vcf_dir` at one of its set directories, e.g.
`data/gtex11/snp/eqtl`.

---

#### Contacts

Dave Kelley (codeowner)
