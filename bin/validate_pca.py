import os 
import numpy as np
from speculator import SpectrumPCA

import matplotlib as mpl 
mpl.use('Agg') 
import matplotlib.pyplot as plt

version = '0.1'

#model = 'nmf' 
#n_pcas = [30, 50, 50, 30] 
#batches = '0_99'

model   = 'burst' 
n_pcas  = [30, 50, 50, 30] 
batches = '0_199'

# number of parameters
if model == 'nmf': 
    # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
    n_param = 10
elif model == 'burst': 
    n_param = 5
else: 
    raise ValueError

if os.environ['machine'] == 'cori': 
    dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 

# read in wavelengths
wave = np.load(os.path.join(dat_dir, 'wave.%s.npy' % model))

wave_bins = [ #(wave >= 1000) & (wave < 2000), 
        (wave >= 2000) & (wave < 3600), 
        (wave >= 3600) & (wave < 5500), 
        (wave >= 5500) & (wave < 7410), 
        (wave >= 7410) & (wave < 60000)
        ]

str_wbins = [# '.w1000_2000', 
        '.w2000_3600', 
        '.w3600_5500', 
        '.w5500_7410', 
        '.w7410_60000' 
        ]

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
    
    fpca = os.path.join(dat_dir, 'fsps.%s.v%s.seed%s%s.pca%i.hdf5' % (model, version, batches, str_wbins[i], n_pcas[i]))
    print('  loading %s' % fpca)
    PCABasis._load_from_file(fpca) 
    PCABases.append(PCABasis)


# read in test parameters and data
ftest       = os.path.join(dat_dir, 'fsps.%s.v%s.lnspectrum.test.npy' % (model, version))
_lnspec_test = np.load(ftest)
print('loading ... %s' % ftest) 

# calculate PCA reconstructed spectra for test set 
lnspec_test, lnspec_recon, waves = [], [], []
for i in range(len(wave_bins)):
    waves.append(wave[wave_bins[i]])

    _spec = _lnspec_test[:,wave_bins[i]]
    lnspec_test.append(_spec)

    normalized_spec = (_spec - PCABases[i].spectrum_shift) / PCABases[i].spectrum_scale 

    # transform to PCA basis and back
    lnspec_pca       = np.dot(normalized_spec, PCABases[i].pca_transform_matrix.T)
    lnspec_recon.append(np.dot(lnspec_pca,
        PCABases[i].pca_transform_matrix)*PCABases[i].spectrum_scale +
        PCABases[i].spectrum_shift)
    
waves = np.concatenate(waves)
lnspec_test  = np.concatenate(lnspec_test, axis=1) 
lnspec_recon = np.concatenate(lnspec_recon, axis=1)

## impose a minima on log(fSED) based on producing 
#lnspec_test     = lnspec_test.clip(-40., None)
#lnspec_recon    = lnspec_recon.clip(-40., None)

for i in range(5): 
    print(lnspec_recon[i,::1000])
    print(lnspec_test[i,::1000])
    print()

# plot the fraction error 
frac_dspectrum = 1. - np.exp(lnspec_recon - lnspec_test) 
frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
        [0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995], axis=0)

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.fill_between(waves, frac_dspectrum_quantiles[0],
        frac_dspectrum_quantiles[6], fc='C0', ec='none', alpha=0.1, label='99.9%')
sub.fill_between(waves, frac_dspectrum_quantiles[1],
        frac_dspectrum_quantiles[5], fc='C0', ec='none', alpha=0.2, label='99%')
sub.fill_between(waves, frac_dspectrum_quantiles[2],
        frac_dspectrum_quantiles[4], fc='C0', ec='none', alpha=0.3, label='95%')
sub.plot(waves, frac_dspectrum_quantiles[3], c='C0', ls='-') 
sub.plot(waves, np.zeros(len(waves)), c='k', ls=':') 
for wave_bin in wave_bins: 
    sub.vlines(wave[wave_bin][-1], -0.1, 0.1, color='k', linestyle=':')

# mark +/- 1%
sub.plot(waves,  0.01 * np.ones(len(waves)), c='k', ls='--', lw=0.5)
sub.plot(waves, -0.01 * np.ones(len(waves)), c='k', ls='--', lw=0.5)

sub.legend(loc='upper right', fontsize=20)
sub.set_xlabel('wavelength', fontsize=25) 
sub.set_xlim(1e3, 1e4)
sub.set_ylabel(r'$(f_{\rm pca} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25) 
sub.set_ylim(-0.03, 0.03) 
fig.savefig('valid_pca.%s.v%s.%s.png' % (model, version, '_'.join([str(np) for np in n_pcas])), bbox_inches='tight') 


# plot CDF of fractional reconstruction error over desi wavelength 
desi_wlim = (waves > 2e3) & (waves > 1e4) 
mean_frac_dspectrum = np.mean(np.abs(1. - np.exp(lnspec_recon[:,desi_wlim] - lnspec_test[:,desi_wlim])), axis=1)
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
fig.savefig('valid_pca.%s.v%s.%s.cdf.png' % (model, version, '_'.join([str(np) for np in n_pcas])), bbox_inches='tight') 

'''
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
sub.set_xlim(1e3, 1e4)
sub.set_ylabel(r'$(\log f_{\rm pca} - \log f_{\rm fsps})/\log f_{\rm fsps}$', fontsize=25) 
sub.set_ylim(-0.03, 0.03) 
fig.savefig('valid_pca.%s.v%s.%s.lnfrac.png' % (model, version, '_'.join([str(np) for np in n_pcas])), bbox_inches='tight') 
'''
