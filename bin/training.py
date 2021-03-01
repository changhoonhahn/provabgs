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


def prior_burst(): 
    ''' prior on burst contribution 
    '''
    return Infer.load_priors([
        Infer.UniformPrior(0., 13.8),                   # uniform priors tburst 
        Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
        Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2.2, 0.4, label='sed')     # uniform priors on dust_index 
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
    if ibatch == 'test': 
        np.random.seed(123456) 
        nspec = 100000 # batch size 
    else: 
        np.random.seed(ibatch) 
        nspec = 10000 # batch size 

    # load priors 
    if name == 'nmf_bases': 
        priors = prior_nmf(4)
    elif name == 'nmfburst': 
        priors = prior_nmfburst() 

    dat_dir='/global/cscratch1/sd/chahah/provabgs/' # hardcoded to NERSC directory 
    if ibatch == 'test': 
        ftheta = os.path.join(dat_dir,
                'fsps.%s.theta.test.npy' % name) 
        fspectrum = os.path.join(dat_dir, 
                'fsps.%s.lnspectrum.test.npy' % name) 
    else: 
        ftheta = os.path.join(dat_dir,
                'fsps.%s.theta.seed%i.npy' % (name, ibatch)) 
        fspectrum = os.path.join(dat_dir, 
                'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) 

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print() 
        print('--- batch %s already exists ---' % str(ibatch))
        print('--- do not overwrite ---')
        print()  
        return None 
    
    # sample prior and transform 
    _thetas = np.array([priors.sample() for i in range(nspec)])
    thetas = priors.transform(_thetas) 

    # load SPS model  
    if name == 'nmf_bases':
        Msps = Models.FSPS_NMF(name='nmf')
    else: 
        Msps = Models.FSPS_NMF(name=name)
    w_fsps, _ = Msps._fsps_nmf(thetas[0]) 

    # wavelength range set to cover the DESI spectra and photometric filter
    # wavelength range 
    wmin, wmax = 2300., 11030.
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    fwave = os.path.join(dat_dir, 'wave_fsps.npy')
    if not os.path.isfile(fwave): # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])
    
    print()  
    print('--- batch %s ---' % str(ibatch))
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


def fsps_burst_prior_samples(ibatch, ncpu=1): 
    ''' run FSPS SPS model and get stellar population luminosity for a single
    burst. This is to build a separate emulator for the burst component 
    '''
    if ibatch == 'test': 
        np.random.seed(123456) 
        nspec = 100000 # batch size 
    else: 
        np.random.seed(ibatch) 
        nspec = 10000 # batch size 

    # load priors 
    name = 'burst'
    priors = prior_burst() 

    dat_dir='/global/cscratch1/sd/chahah/provabgs/' # hardcoded to NERSC directory 
    if ibatch == 'test': 
        ftheta = os.path.join(dat_dir, 'fsps.%s.theta.test.npy' % name) 
        fspectrum = os.path.join(dat_dir, 'fsps.%s.lnspectrum.test.npy' % name) 
    else: 
        ftheta = os.path.join(dat_dir,
                'fsps.%s.theta.seed%i.npy' % (name, ibatch)) 
        fspectrum = os.path.join(dat_dir, 
                'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) 

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print() 
        print('--- batch %s already exists ---' % str(ibatch))
        print('--- do not overwrite ---')
        print()  
        return None 
    
    # sample prior for burst then fit it into nmfburst theta  
    thetas = np.array([priors.sample() for i in range(nspec)])

    # load SPS model  
    Msps = Models.FSPS_NMF(name='nmfburst')
    w_fsps, _ = Msps._fsps_burst(thetas[0]) 

    # wavelength range set to cover the DESI spectra and photometric filter
    # wavelength range 
    wmin, wmax = 2300., 11030.
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)


    fwave = os.path.join(dat_dir, 'wave_fsps.npy')
    if not os.path.isfile(fwave): # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    print()  
    print('--- batch %s ---' % str(ibatch)) 
    # save parameters sampled from prior 
    print('  saving thetas to %s' % ftheta)
    np.save(ftheta, thetas)

    if (ncpu == 1): # run on serial 
        logspectra = []
        for _theta in thetas:
            _, _spectrum = Msps._fsps_burst(_theta)
            logspectra.append(np.log(_spectrum[wlim]))
    else: 
        def _fsps_model_wrapper(theta):
            _, _spectrum = Msps._fsps_burst(theta)
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
    try: 
        ibatch = int(sys.argv[2]) 
    except ValueError: 
        ibatch = sys.argv[2]
    ncpu    = int(sys.argv[3]) 
    
    if name != 'burst': 
        fsps_prior_samples(name, ibatch, ncpu=ncpu)
    else: 
        fsps_burst_prior_samples(ibatch, ncpu=ncpu) 
