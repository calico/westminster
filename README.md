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

*Note:* Change the two lines of code at the top of './env_vars.sh' to your username and local path.

Alternatively, the environment variables can be set manually:
```sh
export WESTMINSTER_DIR=/home/<user_path>/westminster
export PATH=$WESTMINSTER_DIR/src/westminster/scripts:$PATH
export PYTHONPATH=$WESTMINSTER_DIR/src/westminster/scripts:$PYTHONPATH
```

---

#### Contacts

Dave Kelley (codeowner)
