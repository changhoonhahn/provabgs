import os, sys 
import h5py 
import numpy as np
from functools import partial
from multiprocessing.pool import Pool 

from provabgs import util as UT 
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib

i_batch = int(sys.argv[1])
niter   = int(sys.argv[2])
n_cpu   = int(sys.argv[3])

# read preprocessed photometry and spectroscopy of LOWZ targets (see
# lowz.ipynb) 
fh5 = h5py.File('/global/cfs/cdirs/desi/users/chahah/provabgs/lowz/lowz.obs.%iof25.hdf5' % i_batch, 'r')

targetid = fh5['TARGETID'][...].astype(int)

photo_flux = fh5['photo_flux'][...]
photo_ivar = fh5['photo_ivar'][...]

w_obs = fh5['spec_wave'][...]
f_obs = fh5['spec_flux'][...][:,0,:]
i_obs = fh5['spec_ivar'][...][:,0,:]

f_fiber = fh5['fiber_flux'][...]
sigma_f_fiber = fh5['fiber_sigma_flux'][...]

zred = fh5['redshift'][...].flatten()

fh5.close()

ngals = len(zred)
print('%i LOWZ targets in batch %i' % (ngals, i_batch))

# declare SPS model
m_nmf = Models.NMF(burst=True, emulator=True)

# declare flux calibration
m_fluxcalib = FluxCalib.constant_flux_factor

def run_mcmc(igal): 
    fmcmc = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/lowz/posteriors/', 
            'provabgs.%i.hdf5' % targetid[igal])
    if os.path.isfile(fmcmc): 
        # don't overwrite 
        return None 

    if (zred[igal] > 0.6) or (zred[igal] < 0.): 
        return None 
    
    photo_flux_i = np.array(list(photo_flux[igal][0]))
    photo_ivar_i = np.array(list(photo_ivar[igal][0]))

    if np.sum(photo_ivar_i == 0) > 0: 
        # no photometric flux uncertainties.... wtf
        return None 

    # get observations 
    # set prior
    prior = Infer.load_priors([
        Infer.UniformPrior(5., 11.0, label='sed'),
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
    
    # run MCMC
    zeus_chain = desi_mcmc.run(
        wave_obs=w_obs[igal,:],
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
