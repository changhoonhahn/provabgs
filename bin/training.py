'''


script for generating training and testing data for SPS emulator 


'''
import os, sys
import numpy as np 
import multiprocess as mp
from provabgs import infer as Infer
from provabgs import models as Models


def prior_nmf(ncomp): 
    ''' prior on 4 component NMF by Rita
    '''
    return Infer.load_priors([
        Infer.FlatDirichletPrior(ncomp, label='sed'),       # flat dirichilet priors
        Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
        Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2.2, 0.4, label='sed'),     # uniform priors on dust_index 
        Infer.UniformPrior(0., 0.6, label='sed')       # uniformly sample redshift
        ])


def prior_nmfburst(): 
    ''' prior on 4 component NMF by Rita
    '''
    return Infer.load_priors([
        Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
        Infer.UniformPrior(0., 1.),                     # uniform priors fburst
        Infer.UniformPrior(0., 13.8),                   # uniform priors tburst 
        Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
        Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2.2, 0.4, label='sed'),     # uniform priors on dust_index 
        Infer.UniformPrior(0., 0.6, label='sed')       # uniformly sample redshift 
        ])


def fsps_prior_samples(name, ibatch, ncpu=1): 
    ''' run FSPS SPS model and get composite stellar population luminosity.
    '''
    np.random.seed(ibatch) 

    # load priors 
    if name in ['nmf_bases', 'nmf_tng4']: 
        priors = prior_nmf(4)
    elif name == 'nmfburst': 
        priors = prior_nmfburst() 

    nspec = 10000 # batch size 
    
    # sample prior and transform 
    _thetas = np.array([priors.sample() for i in range(nspec)])
    thetas = priors.transform(_thetas) 

    # load SPS model  
    Msps = Models.FSPS(name=name)
    w_fsps, _ = Msps._fsps_nmf(thetas[0]) 

    # wavelength range set to cover the DESI spectra and photometric filter
    # wavelength range 
    wmin, wmax = 2300., 11030.
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    dat_dir='/global/cscratch1/sd/chahah/provabgs/' # hardcoded to NERSC directory 

    fwave = os.path.join(dat_dir, 'wave_fsps.npy')
    if not os.path.isfile(fwave): # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    ftheta = os.path.join(dat_dir,
            'fsps.%s.theta.seed%i.npy' % (name, ibatch)) 
    fspectrum = os.path.join(dat_dir, 
            'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) 

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print() 
        print('--- batch %i already exists ---' % ibatch)
        print('--- do not overwrite ---' % ibatch)
        print()  
        return None 
    else: 
        print()  
        print('--- batch %i ---' % ibatch)
        # save parameters sampled from prior 
        print('  saving thetas to %s' % ftheta)
        np.save(ftheta, thetas)

        if (ncpu == 1): # run on serial 
            logspectra = []
            for _theta in thetas:
                _, _spectrum = Msps._fsps_nmf(_theta)
                logspectra.append(np.log(_spectrum[wlim]))
        else: 
            def _fsps_model_wrapper(theta):
                _, _spectrum = Msps._fsps_nmf(theta)
                return np.log(_spectrum[wlim]) 

            pewl = mp.Pool(ncpu) 
            logspectra = pewl.map(_fsps_model_wrapper, thetas) 

        print('  saving ln(spectra) to %s' % fspectrum)
        np.save(fspectrum, np.array(logspectra))
        print()  
    return None 


if __name__=='__main__': 
    mp.freeze_support()
    name    = sys.argv[1]
    ibatch  = int(sys.argv[2]) 
    ncpu    = int(sys.argv[3]) 

    fsps_prior_samples(name, ibatch, ncpu=ncpu)
