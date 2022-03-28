.. provabgs documentation master file, created by
   sphinx-quickstart on Mon Mar 28 13:54:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The PRObabilistic Value-Added Bright Galaxy Survey (PROVABGS) Catalog
=====================================================================

The PROVABGS catalog will provide measurements of galaxy properties, such as stellar mass, 
star formation rate, stellar metallicity, and stellar age, for >10 million galaxies of the 
|desi|_ Bright Galaxy Survey. 
Full posterior distributions of the galaxy properties will be inferred using state-of-the-art
Bayesian spectral energy distribution (SED) modeling of DESI spectroscopy and photometry.
``provabgs`` provides the full SED modeling pipeline. 
For further details on the PROVABGS SED modeling framework, see |mocha|_.

``provabgs`` pipeline
---------------------
The ``provabgs`` pipeline includes: 

- a state-of-the-art stellar population synthesis (SPS) model based on
  non-parametric prescription for star formation history, a metallicity 
  history that varies over the age of the galaxy, and a flexible dust 
  prescription. 
- a neural network emulator (Kwon *et al.* in prep)  for the SPS model
  that enables accelerated inference. Full posteriors of the 12 SPS parameters 
  can be derived in ~10 minutes. The emulator is currently designed for
  galaxies from 0 < z < 0.6.
- a Bayesian inference based on the |zeus|_ ensemble slice Markov Chain 
  Monte Carlo (MCMC) sampler. 


Catalog Data Releases
---------------------
Coming soon!


Team
----

- |chang|_ (Princeton)
- Rita Tojeiro (St Andrews)
- Justin Alsing (Stockholm) 
- James Kyubin Kwon (Berkeley) 

.. _chang: https://changhoonhahn.github.io/
.. |chang| replace:: ChangHoon Hahn

.. _desi: http://desi.lbl.gov/
.. |desi| replace:: DESI 

.. _zeus: https://github.com/minaskar/zeus
.. |zeus| replace:: ``zeus`` 

.. _mocha: https://arxiv.org/abs/2203.07391
.. |mocha| replace:: Hahn *et al.* (2022b) 


.. toctree::
   :maxdepth: 2

   nersc
