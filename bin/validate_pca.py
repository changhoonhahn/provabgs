import os 
import numpy as np

from speculator import SpectrumPCA
from speculator import Speculator

import matplotlib.pyplot as plt


#model = 'nmf_bases'
model = 'nmfburst'

# number of PCA components 
n_pcas  = [60, 30, 30]

# number of parameters
if model == 'nmf_bases':
    n_param = 10 
elif model == 'nmfburst': 
    n_param = 12 

dat_dir = '/Users/chahah/data/provabgs/'

# read in wavelenght values 
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy'))

wave_bins = [(wave < 4500), (wave >= 4500) & (wave < 6500), (wave >= 6500)]

# read in SpectrumPCA objects
PCABases = []
for i in range(len(n_pcas)):
    # load trained PCA basis object
    PCABasis = SpectrumPCA( 
            n_parameters=n_param,       # number of parameters
            n_wavelengths=np.sum(wave_bins[i]),       # number of wavelength values
            n_pcas=n_pcas[i],              # number of pca coefficients to include in the basis
            spectrum_filenames=None,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=[], # list of filenames containing the corresponding parameter values
            parameter_selection=None) # pass an optional function that takes in parameter vector(s) and returns True/False for any extra parameter cuts we want to impose on the training sample (eg we may want to restrict the parameter ranges)

    fpca = os.path.join(dat_dir, 
            'fsps.%s.seed0_499.w%i.pca%i.hdf5' % (model, i, n_pcas[i]))
    PCABasis._load_from_file(fpca) 
    PCABases.append(PCABasis)


# read in test parameters and data
theta_test      = np.load(os.path.join(dat_dir,
    'fsps.%s.theta.test.npy' % model))
_lnspec_test    = np.load(os.path.join(dat_dir, 
    'fsps.%s.lnspectrum.test.npy' % model))

lnspec_test = []
for wave_bin in wave_bins: 
    lnspec_test.append(_lnspec_test[:,wave_bin])


# calculate PCA reconstructed spectra for test set 
lnspec_recon = []
for i in range(len(wave_bins)):
    _spec = lnspec_test[i]
    normalized_spec = (_spec - PCABases[i].spectrum_shift) / PCABases[i].spectrum_scale 

    # transform to PCA basis and back
    lnspec_pca       = np.dot(normalized_spec, PCABases[i].pca_transform_matrix.T)
    lnspec_recon.append(np.dot(lnspec_pca,
        PCABases[i].pca_transform_matrix)*PCABases[i].spectrum_scale +
        PCABases[i].spectrum_shift)


fig = plt.figure(figsize=(15,5))
for iwave in range(len(wave_bins)): 
    # calculate fraction error 
    frac_dspectrum = 1. - np.exp(lnspec_recon[iwave] - lnspec_test[iwave]) 
    frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
            [0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995], axis=0)

    sub = fig.add_subplot(1,3,iwave+1)
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[0],
            frac_dspectrum_quantiles[6], fc='C0', ec='none', alpha=0.1, label='99.9%')
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[1],
            frac_dspectrum_quantiles[5], fc='C0', ec='none', alpha=0.2, label='99%')
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[2],
            frac_dspectrum_quantiles[4], fc='C0', ec='none', alpha=0.3, label='95%')
    sub.plot(wave[wave_bins[iwave]], frac_dspectrum_quantiles[3], c='C0', ls='-') 
    sub.plot(wave, np.zeros(len(wave)), c='k', ls=':') 

    # mark +/- 1%
    sub.plot(wave, 0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)
    sub.plot(wave, -0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)

    if iwave == len(wave_bins) - 1: sub.legend(loc='upper right', fontsize=20)
    if iwave == 0: 
        sub.set_xlim(2.3e3, 4.5e3)
    elif iwave == 1: 
        sub.set_xlabel('wavelength ($A$)', fontsize=25) 
        sub.set_xlim(4.5e3, 6.5e3)
    elif iwave == 2: 
        sub.set_xlim(6.5e3, 1e4)
    if iwave == 0: sub.set_ylabel(r'$(f_{\rm pca} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25) 
    sub.set_ylim(-0.03, 0.03) 
    if iwave != 0 : sub.set_yticklabels([])

fig.savefig('fsps.%s.valid_pca.%i_%i_%i.png' % (model, n_pcas[0], n_pcas[1], n_pcas[2]), bbox_inches='tight') 
