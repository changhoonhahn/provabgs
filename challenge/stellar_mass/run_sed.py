import os, sys 
import pickle
import numpy as np
from astropy.io import fits 
from astropy.table import Table

from functools import partial
from multiprocessing.pool import Pool 

from provabgs import util as UT 
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib


# declare SPS model
m_nmf = Models.NMF(burst=True, emulator=True)

# declare flux calibration
m_fluxcalib = FluxCalib.constant_flux_factor


def read_spectrophotometry(igal): 
    ''' read spectrophotometry of galaxy igal in bgs-test.fits 
    '''
    test = Table.read('/global/cfs/cdirs/desicollab/science/gqp/stellar_mass_comparison/bgs-test.fits')[igal]

    # redshift 
    zred = test['Z']
    
    # get extinction corrected photometry  
    flux_g = test['FLUX_G'] / test['MW_TRANSMISSION_G'] 
    flux_r = test['FLUX_R'] / test['MW_TRANSMISSION_R'] 
    flux_z = test['FLUX_Z'] / test['MW_TRANSMISSION_Z'] 

    fiberflux_r = test['FIBERFLUX_R'] / test['MW_TRANSMISSION_R'] 

    photo_flux = np.array([flux_g, flux_r, flux_z])
    photo_ivar_flux = np.array([test['FLUX_IVAR_G'], 
                                test['FLUX_IVAR_R'],
                                test['FLUX_IVAR_Z']])

    # fiber flux fraction estimate 
    f_fiber = test['FIBERFLUX_R'] / test['FLUX_R']
    sigma_f_fiber = f_fiber * test['FLUX_IVAR_R']**-0.5

    # read spectrum
    fspec = test['Spectra_Path']
    spec = readDESIspec(fspec)

    is_target = spec['TARGETID'] == test['TARGETID']
    if np.sum(is_target) != 1: raise ValueError 
    # spectra  
    w_obs = np.concatenate([spec['wave_b'], spec['wave_r'], spec['wave_z']])
    f_obs = np.concatenate([spec['flux_b'][is_target][0], 
                            spec['flux_r'][is_target][0], 
                            spec['flux_z'][is_target][0]])
    i_obs = np.concatenate([spec['ivar_b'][is_target][0], 
                            spec['ivar_r'][is_target][0], 
                            spec['ivar_z'][is_target][0]])
    return zred, photo_flux, photo_ivar_flux, f_fiber, sigma_f_fiber, w_obs, f_obs, i_obs 


def readDESIspec(ffits): 
    ''' read DESI spectra fits file

    :params ffits: 
        name of fits file  
    
    :returns spec:
        dictionary of spectra
    '''
    fitobj = fits.open(ffits)
    
    spec = {} 
    for i_k, k in enumerate(['wave', 'flux', 'ivar']): 
        spec[k+'_b'] = fitobj[3+i_k].data
        spec[k+'_r'] = fitobj[8+i_k].data
        spec[k+'_z'] = fitobj[13+i_k].data

    spec['TARGETID'] = fitobj[1].data['TARGETID']
    return spec 


def run_mcmc(igal, niter=3000): 
    fmcmc = os.path.join('/global/cscratch1/sd/chahah/provabgs/challenge/stellar_mass', 
                         'bgs_test.%i.hdf5' % igal) 
    if os.path.isfile(fmcmc): 
        # don't overwrite 
        return None 

    # get observations 
    zred, photo_flux, photo_ivar_flux, f_fiber, sigma_f_fiber, w_obs, f_obs, i_obs =\
            read_spectrophotometry(igal)

    # set prior
    prior = Infer.load_priors([
        Infer.UniformPrior(6., 13, label='sed'),
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

    # run MCMC
    zeus_chain = desi_mcmc.run(
        wave_obs=w_obs,
        flux_obs=f_obs,
        flux_ivar_obs=i_obs,
        bands='desi', # g, r, z
        photo_obs=photo_flux, 
        photo_ivar_obs=photo_ivar_flux, 
        zred=zred,
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


if __name__=='__main__': 
    igal0 = int(sys.argv[1])
    igal1 = int(sys.argv[2])
    niter = int(sys.argv[3])
    n_cpu = int(sys.argv[4])

    pool = Pool(processes=n_cpu) 
    pool.map(partial(run_mcmc), np.arange(igal0, igal1))
    pool.close()
    pool.terminate()
    pool.join()

