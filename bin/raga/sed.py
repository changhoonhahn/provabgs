import os, sys 
import pickle
import numpy as np
from functools import partial
from multiprocessing.pool import Pool 

import sv

from provabgs import util as UT 
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib

i0 = int(sys.argv[1])
i1 = int(sys.argv[2])
sample = sys.argv[3]
niter = int(sys.argv[4])
n_cpu = int(sys.argv[5])


# declare SPS model
m_nmf = Models.NMF(burst=True, emulator=True)

# declare flux calibration
m_fluxcalib = FluxCalib.constant_flux_factor


def run_mcmc(igal): 
    # get observations 
    zred_i, photo_flux_i, photo_ivar_i, w_obs, f_obs, i_obs, f_fiber, sigma_f_fiber\
            = sv.get_spectrophotometry(igal, sample=sample)
    # set prior
    prior = Infer.load_priors([
        Infer.UniformPrior(7., 12.5, label='sed'),
        Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
        Infer.UniformPrior(0., 1., label='sed'), # burst fraction
        Infer.UniformPrior(1e-2, 13.27, label='sed'), # tburst
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2., 1., label='sed'),    # uniform priors on dust_index
        Infer.GaussianPrior(f_fiber, sigma_f_fiber**2, label='flux_calib') # flux calibration
    ])

    desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior, flux_calib=m_fluxcalib)

    fmcmc = os.path.join('/global/cscratch1/sd/chahah/provabgs/raga/', 
            sample.replace('.fits', '.%i.hdf5' % igal))
    # run MCMC
    zeus_chain = desi_mcmc.run(
        wave_obs=w_obs,
        flux_obs=f_obs,
        flux_ivar_obs=i_obs,
        bands='desi', # g, r, z
        photo_obs=photo_flux_i,
        photo_ivar_obs=photo_ivar_i,
        zred=zred_i,
        vdisp=0.,
        sampler='zeus',
        nwalkers=30,
        burnin=0,
        opt_maxiter=2000,
        niter=niter,
        progress=True,
        debug=True,
        writeout=fmcmc,
        overwrite=True)
    return None 

pool = Pool(processes=n_cpu) 
pool.map(partial(run_mcmc), np.arange(i0, i1+1))
pool.close()
pool.terminate()
pool.join()
