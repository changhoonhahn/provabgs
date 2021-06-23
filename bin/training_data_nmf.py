'''

script for generating training and testing data for the 
NMF portion of the SPS emulator 

Notes: 
* 2021/06/21: lower upper limit on metalliicty history; wider wavelength range
    that extends to FUV 
'''
import os, sys
import numpy as np 
import multiprocess as mp
from provabgs import infer as Infer
from provabgs import models as Models

mp.freeze_support()
###########################################################################################
# input 
###########################################################################################
name    = 'nmf' 
version = '0.1'
try: 
    ibatch = int(sys.argv[1]) 
except ValueError: 
    ibatch = sys.argv[1]
    assert ibatch == 'test'
ncpu    = int(sys.argv[2]) 
###########################################################################################

# priors of burst component 
priors = Infer.load_priors([
    Infer.FlatDirichletPrior(4, label='sed'),       # flat dirichilet priors
    Infer.LogUniformPrior(4.5e-5, 2.0e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.LogUniformPrior(4.5e-5, 2.0e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
    Infer.UniformPrior(-3., 1., label='sed'),     # uniform priors on dust_index 
    Infer.UniformPrior(0., 0.6, label='sed')       # uniformly sample redshift
    ])

dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator' # hardcoded to NERSC directory 
if ibatch == 'test': 
    np.random.seed(123456) 
    nspec = 100000 # batch size 
    ftheta = os.path.join(dat_dir, 'fsps.%s.v%s.theta.test.npy' % (name, version)) 
    ftheta_unt = os.path.join(dat_dir, 'fsps.%s.v%s.theta_unt.test.npy' % (name, version)) 
    fspectrum = os.path.join(dat_dir, 'fsps.%s.v%s.lnspectrum.test.npy' % (name, version)) 
else: 
    np.random.seed(ibatch) 
    nspec = 10000 # batch size 
    ftheta = os.path.join(dat_dir, 'fsps.%s.v%s.theta.seed%i.npy' % (name, version, ibatch)) 
    ftheta_unt = os.path.join(dat_dir, 'fsps.%s.v%s.theta_unt.seed%i.npy' % (name, version, ibatch)) 
    fspectrum = os.path.join(dat_dir, 'fsps.%s.v%s.lnspectrum.seed%i.npy' % (name, version, ibatch)) 

# load SPS model  
Msps = Models.NMF(burst=False, emulator=False)
Msps._ssp_initiate()

# sample prior and transform 
_thetas = np.array([priors.transform(priors.sample()) for i in range(nspec)])
zred    = _thetas[:,-1]
tage    = Msps.cosmo.age(zred).value  # age in Gyr
thetas  = np.zeros((nspec, 11))
thetas[:,1:-1]  = _thetas[:,:-1]
thetas[:,-1]    = tage

_thetas_unt = priors.untransform(_thetas)
thetas_unt          = np.zeros((nspec, 10))
thetas_unt[:,0]     = thetas[:,0]
thetas_unt[:,1:-1]  = _thetas_unt[:,:-1]
thetas_unt[:,-1]    = tage

w_fsps = Msps._ssp.wavelengths

# wavelength range set to cover GALEX FUV to WISE W2
wmin, wmax = 1000., 60000.
wlim = (w_fsps >= wmin) & (w_fsps <= wmax)
wave = w_fsps[wlim]

fwave = os.path.join(dat_dir, 'wave.%s.npy' % name)
if not os.path.isfile(fwave): # save FSPS wavelength if not saved 
    np.save(fwave, wave)
    
print()  
print('--- batch %s ---' % str(ibatch)) 
# save parameters sampled from prior 
print('  saving thetas to %s' % ftheta)
np.save(ftheta, thetas)
np.save(ftheta_unt, thetas_unt)

if (ncpu == 1): # run on serial 
    logspectra = []
    for _theta in thetas:
        _, _spectrum = Msps._fsps(_theta[:-1], _theta[-1])
        logspectra.append(np.log(_spectrum[wlim]))
else: 
    def _fsps_model_wrapper(theta):
        _, _spectrum = Msps._fsps(theta[:-1], theta[-1])
        return np.log(_spectrum[wlim]) 

    pewl = mp.Pool(ncpu) 
    logspectra = pewl.map(_fsps_model_wrapper, thetas) 

print('  saving ln(spectra) to %s' % fspectrum)
np.save(fspectrum, np.array(logspectra))
print()  

# divide spectra into 4 wavelength lengths determined by FSPS wavelength
# binning  
# 1. wave  < 3600 (FUV-NUV wavelenght bins dlambda = 20A) 
# 2. 3600 < wave < 5500 (optical wavelength bins dlambda = 0.9)
# 3. 5500 < wave < 7410 (optical wavelength bins dlambda = 0.9)
# 4. 7410 < wave < 59900 (NIR-IR wavelength bins dlambda > 20)
print('  dividing wavelengths')
wave_bin0 = (wave < 3600) 
wave_bin1 = (wave >= 3600) & (wave < 5500) 
wave_bin2 = (wave >= 5500) & (wave < 7410) 
wave_bin3 = (wave >= 7410) 

np.save(fspectrum.replace('.npy', '.w0.npy'), np.array(logspectra)[:,wave_bin0])
np.save(fspectrum.replace('.npy', '.w1.npy'), np.array(logspectra)[:,wave_bin1])
np.save(fspectrum.replace('.npy', '.w2.npy'), np.array(logspectra)[:,wave_bin2])
np.save(fspectrum.replace('.npy', '.w3.npy'), np.array(logspectra)[:,wave_bin3])
