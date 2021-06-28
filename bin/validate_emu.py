'''

validate the PCA NN 

'''
import os, sys 
import pickle
import numpy as np

from provabgs import infer as Infer

from speculator import SpectrumPCA
from speculator import Speculator

import matplotlib as mpl 
mpl.use('Agg') 
import matplotlib.pyplot as plt

#-------------------------------------------------------
# input 
#-------------------------------------------------------
version = '0.1'

model = 'nmf'
archs = ['8x256', '8x256', '8x256', '8x256'] # architectures
n_pcas = [70, 50, 50, 30]
#n_pcas = [50, 50, 50, 30]
n_param = 10 
nbatch = 100 

#model = 'burst'
#archs = ['8x256', '8x256', '8x256', '8x256'] # architectures
#archs = ['10x256', '10x256', '10x256', '10x256'] # architectures
#archs = ['6x512', '6x512', '6x512', '6x512'] # architectures
#n_pcas = [70, 50, 50, 30]
#n_param = 4
#nbatch = 200 

desc = 'nbatch250'
#dat_dir = '/Users/chahah/data/provabgs/'
dat_dir = '/tigress/chhahn/provabgs/emulator/'

# read in wavelenght values 
wave = np.load(os.path.join(dat_dir, 'wave.%s.npy' % model))

wave_bins = [
        (wave < 3600), 
        (wave >= 3600) & (wave < 5500), 
        (wave >= 5500) & (wave < 7410), 
        (wave >= 7410)]


# read in test parameters and data
if model == 'nmf': 
    _theta_test = np.load(os.path.join(dat_dir, 
        'fsps.%s.v%s.theta_unt.test.npy' % (model, version))).astype(np.float32)
    theta_test = _theta_test[:,1:]
    n_param = 9
elif model == 'burst':
    theta_test      = np.load(os.path.join(dat_dir,
        'fsps.%s.v%s.theta.test.npy' % (model, version))).astype(np.float32)
#    # convert Z to log Z 
#    theta_test[:,0] = np.log10(theta_test[:,0])
#    theta_test[:,1] = np.log10(theta_test[:,1])
_lnspec_test    = np.load(os.path.join(dat_dir, 
    'fsps.%s.v%s.lnspectrum.test.npy' % (model, version)))

# get pca values of test data
lnspec_pca_test = []
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
            'fsps.%s.v%s.seed0_%i.w%i.pca%i.hdf5' % (model, version, nbatch-1, i, n_pcas[i]))
    print('  loading %s' % fpca)
    PCABasis._load_from_file(fpca) 

    # calculate PCA reconstructed spectra for test set 
    _spec = _lnspec_test[:,wave_bins[i]]
    normalized_spec = (_spec - PCABasis.spectrum_shift) / PCABasis.spectrum_scale 

    # transform to PCA basis and back
    lnspec_pca_test.append(np.dot(normalized_spec, PCABasis.pca_transform_matrix.T))

lnspec_pca_test = np.concatenate(lnspec_pca_test, axis=1) 

lnspec_test = []
for wave_bin in wave_bins: 
    lnspec_test.append(_lnspec_test[:,wave_bin])
lnspec_test = np.concatenate(lnspec_test, axis=1) 

_wave = [] 
for wave_bin in wave_bins: 
    _wave.append(wave[wave_bin]) 
_wave = np.concatenate(_wave)

# load speculator 
emus = [] 
for i in range(len(wave_bins)): 
    femu = os.path.join(dat_dir, '%s.v%s.seed0_%i.w%i.pca%i.%s.%s' % (model, version, nbatch-1, i, n_pcas[i], archs[i], desc))
    print('   loading %s' % femu)
    _spec = Speculator(restore=True, restore_filename=femu)
    emus.append(_spec)

def emu_lnspec_pca(theta):
    lnspec = [emus[i](theta) for i in range(len(emus))]
    return np.concatenate(lnspec, axis=1)

def emu_lnspec(theta):
    lnspec = [emus[i].log_spectrum(theta) for i in range(len(emus))]
    return np.concatenate(lnspec, axis=1)

# reconstructed ln(spec) 
lnspec_pca_recon = emu_lnspec_pca(theta_test) 
lnspec_recon = emu_lnspec(theta_test) 

for i in range(int(5e2))[::100]: 
    print(lnspec_pca_recon[i,::100])
    print(lnspec_pca_test[i,::100])
'''
    frac_dpca = 1. - (lnspec_pca_recon / lnspec_pca_test)
    frac_dpca_quantiles = np.nanquantile(frac_dpca, 
            [0.0005, 0.005, 0.025, 0.16, 0.5, 0.84, 0.975, 0.995, 0.9995], axis=0)

    # plot fractional error on predicted PCA coefficients 
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    sub.fill_between(range(lnspec_pca_test.shape[1]), frac_dpca_quantiles[0], frac_dpca_quantiles[-1], 
            fc='C0', ec='none', alpha=0.1, label='99.9%')
    sub.fill_between(range(lnspec_pca_test.shape[1]), frac_dpca_quantiles[1], frac_dpca_quantiles[-2], 
            fc='C0', ec='none', alpha=0.2, label='99%')
    sub.fill_between(range(lnspec_pca_test.shape[1]), frac_dpca_quantiles[2], frac_dpca_quantiles[-3], 
            fc='C0', ec='none', alpha=0.3, label='95%')
    sub.fill_between(range(lnspec_pca_test.shape[1]), frac_dpca_quantiles[3], frac_dpca_quantiles[-4],
            fc='C0', ec='none', alpha=0.5, label='68%')
    sub.plot(range(lnspec_pca_test.shape[1]), frac_dpca_quantiles[4], c='C0', ls='-') 
    sub.plot(range(lnspec_pca_test.shape[1]), np.zeros(lnspec_pca_test.shape[1]), c='k', ls=':') 

    # mark +/- 1%
    sub.plot(range(lnspec_pca_test.shape[1]),  0.01 * np.ones(lnspec_pca_test.shape[1]), c='k', ls='--', lw=0.5)
    sub.plot(range(lnspec_pca_test.shape[1]), -0.01 * np.ones(lnspec_pca_test.shape[1]), c='k', ls='--', lw=0.5)

    sub.legend(loc='upper right', fontsize=20)
    sub.set_xlabel('PCA components', fontsize=25) 
    sub.set_xlim(0., lnspec_pca_test.shape[1]) 
    sub.set_ylabel(r'$(\alpha_{\rm emu} - \alpha_{\rm fsps})/\alpha_{\rm fsps}$', fontsize=25) 
    sub.set_ylim(-0.5, 0.5) 
    fig.savefig(, bbox_inches='tight') 
'''

# plot fractional error
fig = plt.figure(figsize=(15,5))
# calculate fraction error 
frac_dspectrum = 1. - np.exp(lnspec_recon - lnspec_test) 
frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
        [0.0005, 0.005, 0.025, 0.16, 0.5, 0.84, 0.975, 0.995, 0.9995], axis=0)

sub = fig.add_subplot(111)
sub.fill_between(_wave, frac_dspectrum_quantiles[0], frac_dspectrum_quantiles[-1], 
        fc='C0', ec='none', alpha=0.1, label='99.9%')
sub.fill_between(_wave, frac_dspectrum_quantiles[1], frac_dspectrum_quantiles[-2], 
        fc='C0', ec='none', alpha=0.2, label='99%')
sub.fill_between(_wave, frac_dspectrum_quantiles[2], frac_dspectrum_quantiles[-3], 
        fc='C0', ec='none', alpha=0.3, label='95%')
sub.fill_between(_wave, frac_dspectrum_quantiles[3], frac_dspectrum_quantiles[-4],
        fc='C0', ec='none', alpha=0.5, label='68%')
sub.plot(_wave, frac_dspectrum_quantiles[4], c='C0', ls='-') 
sub.plot(wave, np.zeros(len(wave)), c='k', ls=':') 

# mark +/- 1%
sub.plot(wave, 0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)
sub.plot(wave, -0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)

sub.legend(loc='upper right', fontsize=20)
sub.set_xlabel('wavelength ($A$)', fontsize=25) 
sub.set_xlim(2.3e3, 1e4) 
sub.set_ylabel(r'$(f_{\rm emu} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25) 
sub.set_ylim(-0.1, 0.1) 

ffig = 'valid_emu.%s.v%s.%s.%s.%s.png' % (model, version, '_'.join(str(nn) for nn in n_pcas), '.'.join(archs), desc)
fig.savefig(ffig, bbox_inches='tight') 

# plot cumulative fractional error
desi_wlim = (_wave > 2e3) & (_wave < 1e4) # over desi wavelenght
mean_frac_dspectrum = np.mean(np.abs(frac_dspectrum[:,desi_wlim]), axis=1)
quant = np.quantile(mean_frac_dspectrum, [0.68, 0.95, 0.99, 0.999])

fig = plt.figure(figsize=(8,6))
sub = fig.add_subplot(111)
for q, a in zip(quant[::-1], [0.1, 0.2, 0.3, 0.5]):
  sub.fill_between([0., q], [0., 0.], [1., 1.], alpha=a, color='C0')
_ = sub.hist(mean_frac_dspectrum, 40, range=(0., 0.05), density=True, histtype='step', cumulative=True, color='k')
sub.set_xlabel(r'${\rm mean}_\lambda \langle (f_{\rm emu}  - f_{\rm fsps}) / f_{\rm fsps} \rangle$', fontsize=20)
sub.set_xlim(0., 0.03)
sub.set_ylabel('cumulative distribution', fontsize=20)
sub.set_ylim(0., 1.)
fig.savefig(ffig.replace('.png', '.cdf.png'), bbox_inches='tight') 

outlier = mean_frac_dspectrum > 0.1
print('%i with frac err > 0.1' % np.sum(outlier))
# plot outliers 
fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
for i in range(np.sum(outlier)): 
    sub.plot(_wave, np.exp(lnspec_test[outlier][i]))
    sub.plot(_wave, np.exp(lnspec_recon[outlier][i]), c='k', ls=':')
sub.set_xlabel('wavelength ($A$)', fontsize=25) 
sub.set_xlim(2.3e3, 1e4) 
fig.savefig(ffig.replace('.png', '.outliers.png'), bbox_inches='tight') 
'''
# plot loss 
losses = [np.loadtxt(os.path.join(dat_dir, '%s.v%s.seed0_%i.w%i.pca%i.%s.%s.loss.dat' % (model, version, nbatch-1, i, n_pcas[i], archs[i], desc))) for i in range(len(wave_bins))]

fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
for i, loss in enumerate(losses): 
    sub.plot(np.arange(loss.shape[0]), loss[:,2], label='wave bin %i' % i)

sub.legend(loc='upper right')
sub.set_ylabel('loss', fontsize=25)
sub.set_yscale('log')
sub.set_xlabel('Epochs', fontsize=25)
sub.set_xlim(0, loss.shape[0])
fig.savefig(ffig.replace('.png', '.loss.png')
'''
