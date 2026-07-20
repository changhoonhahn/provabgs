# PRObabilistic Value-Added Bright Galaxy Survey (PROVABGS)
[![CI](https://github.com/changhoonhahn/provabgs/actions/workflows/ci.yml/badge.svg)](https://github.com/changhoonhahn/provabgs/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2202.01809-b31b1b.svg)](https://arxiv.org/abs/2202.01809)
[![arXiv](https://img.shields.io/badge/arXiv-2209.14323-b31b1b.svg)](https://arxiv.org/abs/2209.14323)

The PROVABGS catalog will provide measurements of galaxy properties, such as stellar mass, 
star formation rate, stellar metallicity, and stellar age for >10 million galaxies of the 
[DESI](http://desi.lbl.gov/) Bright Galaxy Survey. 
Full posterior distributions of these galaxy properties will be inferred using state-of-the-art
Bayesian spectral energy distribution (SED) modeling of DESI spectroscopy and photometry.

The `provabgs` Python package provides 

- a state-of-the-art stellar population synthesis (SPS) model based on
  non-parametric prescription for star formation history, a metallicity 
  history that varies over the age of the galaxy, and a flexible dust 
  prescription. 
- a neural network emulator (Kwon *et al.* in prep) for the SPS model
  that is >100x faster than the original SPS model and enables accelerated 
  inference. Full posteriors of the 12 SPS parameters can be derived in 
  ~10 minutes. The emulator is currently designed for galaxies from 0 < z < 0.6.
- a Bayesian inference pipeline based on the [zeus](https://github.com/minaskar/zeus)
  ensemble slice Markov Chain Monte Carlo (MCMC) sample.  

For additional details see [documentation](https://changhoonhahn.github.io/provabgs) 
and [Hahn *et al* (2022)](https://arxiv.org/abs/2202.01809)

## Installation
Clone the repo first:
```bash
git clone https://github.com/changhoonhahn/provabgs.git
cd provabgs
```

### Option A: just the neural network emulators (recommended for most users)
This is all you need to evaluate SEDs and run inference with the emulator --
no compiled dependencies (no `fsps`, no `torch`), and it's what the
[example notebook](nb/example.ipynb) uses.
```bash
pip install -e .
```
To also use the Bayesian MCMC fitting pipeline (`provabgs.infer`), add the `mcmc` extra:
```bash
pip install -e ".[mcmc]"
```

### Option B: everything, including the original (non-emulator) FSPS model
`python-fsps` requires compiling FSPS's Fortran code, which is where most
install trouble comes from. Installing via `conda`/`mamba` pulls prebuilt
binaries and avoids that entirely:
```bash
conda env create -f environment.yml
conda activate provabgs
```
If you'd rather use plain `pip`, see the `python-fsps`
[documentation](https://python-fsps.readthedocs.io/en/latest/) for
build instructions, then run `pip install -e ".[full]"`.

### NERSC / cluster notes
If you're using `provabgs` on NERSC, you may run into an import error with
`libgfortran.so.5` when using `fsps`. The following has resolved it before:
```bash
module unload PrgEnv-intel
module load PrgEnv-gnu
```

## Example
Checkout the [nb/example.ipybn](https://github.com/changhoonhahn/provabgs/blob/main/nb/example.ipynb) 
notebook for an example on conducting Bayesian SED modeling on galaxy 
spectra using `provabgs`. It requires less than 10 lines of code and 
about 10 minutes!

If you're interested in conducting Bayesian SED modeling on DESI spectra in
particular, check out the [nb/tutorial_desispec.ipynb](https://github.com/changhoonhahn/provabgs/blob/main/nb/tutorial_desispec.ipynb)
notebook. 

## Development
If you're contributing notebooks or code, please install [`pre-commit`](https://pre-commit.com)
so exploratory/debugging notebooks get their outputs stripped before they're committed:
```bash
pip install pre-commit
pre-commit install
```
This keeps large embedded images/data out of git history (the two tutorial notebooks under
`nb/` are exempt so their rendered outputs stay visible). Please also avoid committing large
binary files (trained model weights, data files) directly -- attach them to a GitHub release
or a Zenodo record and download them instead.

## Team
- [ChangHoon Hahn](https://changhoonhahn.github.io) (Princeton)
- Rita Tojeiro (St Andrews)
- Justin Alsing (Stockholm) 
- James Kyubin Kwon (Berkeley) 


## Contact
If you have any questions or need help using the package, please raise a github issue or contact me at changhoon.hahn@princeton.edu
