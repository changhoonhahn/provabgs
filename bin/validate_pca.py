import os 
import numpy as np

from speculator import SpectrumPCA
from speculator import Speculator

import matplotlib as mpl 
mpl.use('Agg') 
import matplotlib.pyplot as plt

#model = 'nmf' 
#n_pcas = [30, 50, 50, 30] 
#batches = '0_99'

model = 'burst' 
n_pcas = [30, 50, 50, 30] 
batches = '0_199'

#model = 'nmf_bases'
#n_pcas = [50, 30, 30] 
#n_pcas  = [30, 30, 30, 30, 30, 30]

#model = 'nmfburst'
#n_pcas  = [30, 60, 30, 30, 30, 30]

# number of parameters
if model == 'nmf_bases':
    n_param = 10 
elif model == 'nmfburst': 
    n_param = 12 
elif model == 'nmf': 
    # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
    n_param = 10
elif model == 'burst': 
    n_param = 5
else: 
    raise ValueError

if os.environ['machine'] == 'cori': 
    dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 

# read in wavelenght values 
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy'))

wave_bins = [(wave < 3600), 
        (wave >= 3600) & (wave < 5500), 
        (wave >= 5500) & (wave < 7410), 
        (wave >= 7410)]

# read in SpectrumPCA objects
PCABases = []
for i in range(len(n_pcas)):
    n_wave = np.sum(wave_bins[i]) 
    # load trained PCA basis object
    PCABasis = SpectrumPCA( 
            n_parameters=n_param,       # number of parameters
            n_wavelengths=n_wave,       # number of wavelength values
            n_pcas=n_pcas[i],              # number of pca coefficients to include in the basis
            spectrum_filenames=None,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=[], # list of filenames containing the corresponding parameter values
            parameter_selection=None) # pass an optional function that takes in parameter vector(s) and returns True/False for any extra parameter cuts we want to impose on the training sample (eg we may want to restrict the parameter ranges)
    
    fpca = os.path.join(dat_dir, 
            'fsps.%s.seed%s.w%i.pca%i.hdf5' % (model, batches, i, n_pcas[i]))
    print('  loading %s' % fpca)
    PCABasis._load_from_file(fpca) 
    PCABases.append(PCABasis)


# read in test parameters and data
ftest = os.path.join(dat_dir, 'fsps.%s.lnspectrum.test.npy' % model)
lnspec_test    = np.load(ftest)
print('loading ... %s' % ftest) 


# calculate PCA reconstructed spectra for test set 
lnspec_recon = []
for i in range(len(wave_bins)):
    _spec = lnspec_test[:,wave_bins[i]]
    normalized_spec = (_spec - PCABases[i].spectrum_shift) / PCABases[i].spectrum_scale 

    # transform to PCA basis and back
    lnspec_pca       = np.dot(normalized_spec, PCABases[i].pca_transform_matrix.T)
    lnspec_recon.append(np.dot(lnspec_pca,
        PCABases[i].pca_transform_matrix)*PCABases[i].spectrum_scale +
        PCABases[i].spectrum_shift)

lnspec_recon = np.concatenate(lnspec_recon, axis=1)
for i in range(5): 
    print(lnspec_recon[i,::1000])
    print(lnspec_test[i,::1000])
    print()


# plot the fraction error  on ln(spec)
frac_dspectrum = 1. - lnspec_recon / lnspec_test 
print(frac_dspectrum[::1000])
frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
        [0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995], axis=0)

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.fill_between(wave, frac_dspectrum_quantiles[0],
        frac_dspectrum_quantiles[6], fc='C0', ec='none', alpha=0.1, label='99.9%')
sub.fill_between(wave, frac_dspectrum_quantiles[1],
        frac_dspectrum_quantiles[5], fc='C0', ec='none', alpha=0.2, label='99%')
sub.fill_between(wave, frac_dspectrum_quantiles[2],
        frac_dspectrum_quantiles[4], fc='C0', ec='none', alpha=0.3, label='95%')
sub.plot(wave, frac_dspectrum_quantiles[3], c='C0', ls='-') 
sub.plot(wave, np.zeros(len(wave)), c='k', ls=':') 
for wave_bin in wave_bins: 
    sub.vlines(wave[wave_bin][-1], -0.1, 0.1, color='k', linestyle=':')

# mark +/- 1%
sub.plot(wave, 0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)
sub.plot(wave, -0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)

sub.legend(loc='upper right', fontsize=20)
sub.set_xlabel('wavelength', fontsize=25) 
sub.set_xlim(2.3e3, 1e4)
sub.set_ylabel(r'$(\log f_{\rm pca} - \log f_{\rm fsps})/\log f_{\rm fsps}$', fontsize=25) 
sub.set_ylim(-0.03, 0.03) 

fig.savefig('fsps.%s.valid_pca.%s.lnfrac.png' % (model, '_'.join([str(np) for np in n_pcas])), bbox_inches='tight') 


# plot the fraction error 
frac_dspectrum = 1. - np.exp(lnspec_recon - lnspec_test) 
print(frac_dspectrum[::1000])
frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
        [0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995], axis=0)

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.fill_between(wave, frac_dspectrum_quantiles[0],
        frac_dspectrum_quantiles[6], fc='C0', ec='none', alpha=0.1, label='99.9%')
sub.fill_between(wave, frac_dspectrum_quantiles[1],
        frac_dspectrum_quantiles[5], fc='C0', ec='none', alpha=0.2, label='99%')
sub.fill_between(wave, frac_dspectrum_quantiles[2],
        frac_dspectrum_quantiles[4], fc='C0', ec='none', alpha=0.3, label='95%')
sub.plot(wave, frac_dspectrum_quantiles[3], c='C0', ls='-') 
sub.plot(wave, np.zeros(len(wave)), c='k', ls=':') 
for wave_bin in wave_bins: 
    sub.vlines(wave[wave_bin][-1], -0.1, 0.1, color='k', linestyle=':')

# mark +/- 1%
sub.plot(wave, 0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)
sub.plot(wave, -0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)

sub.legend(loc='upper right', fontsize=20)
sub.set_xlabel('wavelength', fontsize=25) 
sub.set_xlim(2.3e3, 1e4)
sub.set_ylabel(r'$(f_{\rm pca} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25) 
sub.set_ylim(-0.03, 0.03) 

fig.savefig('fsps.%s.valid_pca.%s.png' % (model, '_'.join([str(np) for np in n_pcas])), bbox_inches='tight') 


# plot CDF of fractional reconstruction error
mean_frac_dspectrum = np.mean(np.abs(1. - np.exp(lnspec_recon - lnspec_test)), axis=1)
quant = np.quantile(mean_frac_dspectrum, [0.68, 0.95, 0.99, 0.999])
fig = plt.figure(figsize=(8,6))
sub = fig.add_subplot(111)
for q, a in zip(quant[::-1], [0.1, 0.2, 0.3, 0.5]): 
    sub.fill_between([0., q], [0., 0.], [1., 1.], alpha=a, color='C0')
_ = sub.hist(mean_frac_dspectrum, 40, density=True, histtype='step', cumulative=True, color='k')
sub.set_xlabel(r'${\rm mean}_\lambda \langle (f_{\rm speculator}  - f_{\rm fsps}) / f_{\rm fsps} \rangle$', fontsize=20)
sub.set_xlim(0., 0.03)
sub.set_ylabel('cumulative distribution', fontsize=20)
sub.set_ylim(0., 1.)
fig.savefig('fsps.%s.valid_pca.%s.cdf.png' % (model, '_'.join([str(np) for np in n_pcas])), bbox_inches='tight') 
