# PRObabilistic Value-Added Bright Galaxy Survey (PROVABGS)
[![Gitter](https://badges.gitter.im/provabgs/provabgs.svg)](https://gitter.im/provabgs/provabgs?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

`provabgs` is a python package for fitting photometry and spectra from the Dark
Energy Spectroscopic Instrument Bright Galaxy Survey (DESI BGS). 

## Installation
To install the package, clone the github repo and use `pip` to install  
```bash
# clone github repo 
git clone https://github.com/changhoonhahn/provabgs.git

cd provabgs

# install 
pip install -e . 
```

`pip` install coming soon...

### requirements
If you only use the emulators, `provabgs` can run without `fsps`. However, it's
recommended that you install `python-fsps`. See `python-fsps`
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
