# PRObabilistic Value-Added Bright Galaxy Survey (PROVABGS)
[![Gitter](https://badges.gitter.im/provabgs/provabgs.svg)](https://gitter.im/provabgs/provabgs?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

The PROVABGS catalog will provide measurements of galaxy properties, such as stellar mass, 
star formation rate, stellar metallicity, and stellar age for >10 million galaxies of the 
[DESI](http://desi.lbl.gov/) Bright Galaxy Survey. 
Full posterior distributions of the galaxy properties will be inferred using state-of-the-art
Bayesian spectral energy distribution (SED) modeling of DESI spectroscopy and photometry.
`provabgs` provides the full SED modeling pipeline. 

`provabgs` includes: 
- a state-of-the-art stellar population synthesis (SPS) model based on
  non-parametric prescription for star formation history, a metallicity 
  history that varies over the age of the galaxy, and a flexible dust 
  prescription. 
- a Bayesian inference pipeline based on the [zeus](https://github.com/minaskar/zeus)
  ensemble slice Markov Chain Monte Carlo (MCMC) sample.  
- a neural network emulator (Kwon *et al.* in prep)  for the SPS model
  that enables accelerated inference. Full posteriors of the 12 SPS parameters 
  can be derived in ~10 minutes. (The emulator only works for galaxies from 0 <
  z < 0.5)

For further details on the PROVABGS SED modeling framework, see [Hahn *et al* (2022)](https://arxiv.org/abs/2202.01809)

## Installation
To install the package, clone the github repo and use `pip` to install  
```bash
# clone github repo 
git clone https://github.com/changhoonhahn/provabgs.git

cd provabgs

# install 
pip install -e . 
```

### requirements
If you only plan to use `provabgs` with the neural emulators, then `provabgs` 
does not require `fsps`. However, if you want to use the original SPS model, 
you will need to install `python-fsps`.  See `python-fsps`
[documentation](https://python-fsps.readthedocs.io/en/latest/) for installation
instruction. 

If you're using `provabgs` on NERSC, see [below](#fsps-on-nersc) for 
some notes on installing `FSPS` on `NERSC`.

### fsps on NERSC
I've been running into some issues installing and using `fsps` on NERSC. *e.g.*
there's an import error with libgfotran.so.5. The following may resolve the problem... 
```
module unload PrgEnv-intel
module load PrgEnv-gnu

```

## Team
- ChangHoon Hahn (Princeton)
- Rita Tojeiro (St Andrews)
- Justin Alsing (Stockholm) 
- James Kyubin Kwon (Berkeley) 


## Contact
If you have any questions or need help using the package, please raise a github issue, post a message on gitter, or contact me at changhoon.hahn@princeton.edu
