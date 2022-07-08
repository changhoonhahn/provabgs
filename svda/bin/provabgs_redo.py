'''

script for submitting provabgs runs for galaxies specified by 
(healpix, targetid). These are galaxies with bad goodness-of-fits 

'''
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

target  = sys.argv[1]
survey  = sys.argv[2]
niter = int(sys.argv[3])
n_cpu = int(sys.argv[4])

# real healpixs and targetids
hpixels, targetids = np.loadtxt('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/sv3-bright-BGS_BRIGHT.flagged.20220701.dat', unpack=True, usecols=[0,1])

# compile obs
zred, photo_flux, photo_ivar, w_obs, f_obs, i_obs, f_fiber, sigma_f_fiber = [], [], [], [], [], [], [], []
for hpixel, targetid in zip(hpixels, targetids): 
    meta, _zred, _photo_flux, _photo_ivar, _w_obs, _f_obs, _i_obs, _f_fiber, _sigma_f_fiber\
            = SVDA.healpix(hpixel, target=target, redux='fuji', survey=survey)

    is_target = (meta['TARGETID'] == targetid)
    assert np.sum(is_target) == 1 
    
    zred.append(_zred[is_target]) 
    photo_flux.append(_photo_flux[is_target]) 
    photo_ivar.append(_photo_ivar[is_target])
    w_obs.append(_w_obs[is_target])
    f_obs.append(_f_obs[is_target]) 
    i_obs.append(_i_obs[is_target]) 
    f_fiber.append(_f_fiber[is_target])
    sigma_f_fiber.append(_sigma_f_fiber[is_target])


ngals = len(zred)
print('running PROVABGS on %i %s targets' % (ngals, target))

# declare SPS model
m_nmf = Models.NMF(burst=True, emulator=True)

# declare flux calibration
m_fluxcalib = FluxCalib.constant_flux_factor

def run_mcmc(igal): 
    hpix        = hpixels[igal]
    targetid    = targetids[igal] 

    fmcmc = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/healpix/',
            str(hpix), 'provabgs.%i.hdf5' % targetid) 

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
        flux_obs=f_obs[igal],
        flux_ivar_obs=i_obs[igal],
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
