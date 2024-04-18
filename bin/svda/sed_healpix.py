import os, sys 
import pickle
import numpy as np
from functools import partial
from multiprocessing.pool import Pool 

import svda as SVDA

from provabgs import util as UT 
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib


hpix    = int(sys.argv[1])
target  = sys.argv[2]
survey  = sys.argv[3]
niter = int(sys.argv[4])
n_cpu = int(sys.argv[5])

# read BGS targets from specified healpix
meta, zred, photo_flux, photo_ivar, w_obs, f_obs, i_obs, f_fiber, sigma_f_fiber\
        = SVDA.healpix(hpix, target=target, redux='fuji', survey=survey)
ngals = len(meta)
print('%i %s targets in HEALPIX=%i' % (ngals, target, hpix))

# declare SPS model
m_nmf = Models.NMF(burst=True, emulator=True)

# declare flux calibration
m_fluxcalib = FluxCalib.constant_flux_factor

def run_mcmc(igal): 
    fmcmc = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/healpix/',
            str(hpix), 'provabgs.%i.hdf5' % meta['TARGETID'][igal])
    if os.path.isfile(fmcmc): 
        # don't overwrite 
        return None 

    # get observations 
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
        Infer.GaussianPrior(f_fiber[igal], sigma_f_fiber[igal]**2, label='flux_calib') # flux calibration
    ])

    desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior, flux_calib=m_fluxcalib)
    
    photo_flux_i = np.array(list(photo_flux[igal]))
    photo_ivar_i = np.array(list(photo_ivar[igal]))
    
    if (zred[igal] > 0.6) or (zred[igal] < 0.): 
        return None 

    # run MCMC
    zeus_chain = desi_mcmc.run(
        wave_obs=w_obs,
        flux_obs=f_obs[igal,:],
        flux_ivar_obs=i_obs[igal,:],
        bands='desi', # g, r, z
        photo_obs=photo_flux_i, 
        photo_ivar_obs=photo_ivar_i, 
        zred=zred[igal],
        vdisp=0.,
        sampler='zeus',
        nwalkers=30,
        burnin=0,
        opt_maxiter=2000,
        niter=niter,
        progress=False,
        debug=True,
        writeout=fmcmc,
        overwrite=True)
    return None 

pool = Pool(processes=n_cpu) 
pool.map(partial(run_mcmc), np.arange(ngals))
pool.close()
pool.terminate()
pool.join()
