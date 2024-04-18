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


if __name__=='__main__': 
    mp.freeze_support()
    name    = sys.argv[1]
    ibatch0 = int(sys.argv[2]) 
    ibatch1 = int(sys.argv[3]) 
    ncpu    = int(sys.argv[4]) 

    # load priors 
    name = 'burst'
    priors = prior_burst() 
    nspec = 10000 # batch size 
    
    if os.environ['machine'] in ['della', 'tiger']: 
        dat_dir='/tigress/chhahn/provabgs/'
    else: 
        dat_dir='/global/cscratch1/sd/chahah/provabgs/' # hardcoded to NERSC directory 

    fwave = os.path.join(dat_dir, 'wave_fsps.npy')
    if not os.path.isfile(fwave): # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    
    for ibatch in range(ibatch0, ibatch1+1): 

        ftheta = os.path.join(dat_dir,
                'fsps.%s.theta.seed%i.npy' % (name, ibatch)) 
        fspectrum = os.path.join(dat_dir, 
                'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) 

        if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
            print() 
            print('--- batch %s already exists ---' % str(ibatch))
            print('--- do not overwrite ---')
            print()  
            continue 

        np.random.seed(ibatch) 
    
        # sample prior for burst then fit it into nmfburst theta  
        thetas = np.array([priors.sample() for i in range(nspec)])
        thetas = np.concatenate([np.zeros((nspec, 6)), thetas], axis=1) 
        
        if ibatch == ibatch0: 
            # load SPS model  
            Msps = Models.NMF(burst=True, emulator=False)
            w_fsps, _ = Msps._fsps_burst(thetas[0]) 

            # wavelength range set to cover the DESI spectra and photometric filter
            # wavelength range 
            wmin, wmax = 2300., 11030.
            wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

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
