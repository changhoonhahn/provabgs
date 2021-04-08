# PRObabilistic Value-Added Bright Galaxy Survey (PROVABGS)
`provabgs` is a python package for fitting photometry and spectra from the Dark
Energy Spectroscopic Instrument Bright Galaxy Survey (DESI BGS). 


## Team
- ChangHoon Hahn (Princeton)
- Rita Tojeiro (St Andrews)
- Justin Alsing (Stockholm) 
- James Kyubin Kwon (Berkeley) 


## Troubleshooting
#### LLVMLITE
If you encounter the following error message when you `pip install -e .` in the repo directory:  

`ERROR: Failed building wheel for llvmlite`

your `pip` and `llvmlite` might have incompatible versions. Try setting up a new conda environment and run `conda install pip=19.0`, then use the installed `pip` to setup `provabgs`. 


## Contact
If you have any questions or need help using the package, feel free to raise a github issue or contact me at changhoon.hahn@princeton.edu

