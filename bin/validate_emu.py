
import os 
import pickle
import numpy as np

from speculator import SpectrumPCA
from speculator import Speculator

import matplotlib.pyplot as plt


#model = 'nmf_bases'
model = 'nmfburst'

# number of PCA components 
n_pcas  = [50, 30, 30]

# architectures
archs = ['4x256.', '6x256.', '4x256.']

# number of parameters
if model == 'nmf_bases':
    n_param = 10 
elif model == 'nmfburst': 
    n_param = 12 

dat_dir = '/Users/chahah/data/provabgs/'

# read in wavelenght values 
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy'))

wave_bins = [(wave < 4500), (wave >= 4500) & (wave < 6500), (wave >= 6500)]


# plot loss 
losses = [
        np.loadtxt(os.path.join(dat_dir,
            'fsps.%s.seed0_499.w%i.pca%i.%sloss.dat' % (model, i, n_pcas[i],
                archs[i]))) for i in range(3)
            ]

fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
for i, loss in enumerate(losses): 
    sub.plot(np.arange(loss.shape[0]), loss[:,2], label='wave bin %i' % i)

sub.legend(loc='upper right')
sub.set_ylabel('loss', fontsize=25)
sub.set_yscale('log')
sub.set_xlabel('Epochs', fontsize=25)
sub.set_xlim(0, loss.shape[0])
fig.savefig('fsps.%s.%svalid_emu.loss.png' % (model, ''.join(archs)), bbox_inches='tight') 


# read in test parameters and data
theta_test      = np.load(os.path.join(dat_dir,
    'fsps.%s.theta.test.npy' % model)).astype(np.float32)
_lnspec_test    = np.load(os.path.join(dat_dir, 
    'fsps.%s.lnspectrum.test.npy' % model))

lnspec_test = []
for wave_bin in wave_bins: 
    lnspec_test.append(_lnspec_test[:,wave_bin])
lnspec_test = np.concatenate(lnspec_test, axis=1) 


# load speculator 
emus = [
        Speculator(
            restore=True, 
            restore_filename=os.path.join(dat_dir, '_fsps.%s.seed0_499.w%i.pca%i.%slog'% (model, i, n_pcas[i], archs[i]))
            )
        for i in range(3)
        ]

def emu_lnspec(theta):
    lnspec = [emus[i].log_spectrum(theta) for i in range(3)]
    return np.concatenate(lnspec, axis=1)

# reconstructed ln(spec) 
lnspec_recon = emu_lnspec(theta_test) 

# plot fractional error
fig = plt.figure(figsize=(15,5))
for iwave in range(3): 
    # calculate fraction error 
    frac_dspectrum = 1. - np.exp(lnspec_recon[:,wave_bins[iwave]] - lnspec_test[:,wave_bins[iwave]]) 
    frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
            [0.0005, 0.005, 0.025, 0.16, 0.5, 0.84, 0.975, 0.995, 0.9995], axis=0)

    sub = fig.add_subplot(1,3,iwave+1)
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[0],
            frac_dspectrum_quantiles[-1], fc='C0', ec='none', alpha=0.1, label='99.9%')
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[1],
            frac_dspectrum_quantiles[-2], fc='C0', ec='none', alpha=0.2, label='99%')
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[2],
            frac_dspectrum_quantiles[-3], fc='C0', ec='none', alpha=0.3, label='95%')
    sub.fill_between(wave[wave_bins[iwave]], frac_dspectrum_quantiles[3],
            frac_dspectrum_quantiles[-4], fc='C0', ec='none', alpha=0.5, label='68%')
    sub.plot(wave[wave_bins[iwave]], frac_dspectrum_quantiles[4], c='C0', ls='-') 
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
    if iwave == 0: sub.set_ylabel(r'$(f_{\rm emu} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25) 
    sub.set_ylim(-0.03, 0.03) 
    if iwave != 0 : sub.set_yticklabels([])
fig.savefig('fsps.%s.%svalid_emu.png' % (model, ''.join(archs)), bbox_inches='tight') 

# plot cumulative fractional error
mean_frac_dspectrum = np.mean(np.abs(1. - np.exp(lnspec_recon - lnspec_test)), axis=1)
quant = np.quantile(mean_frac_dspectrum, [0.68, 0.95, 0.99, 0.999])

fig = plt.figure(figsize=(8,6))
sub = fig.add_subplot(111)
for q, a in zip(quant[::-1], [0.1, 0.2, 0.3, 0.5]):
  sub.fill_between([0., q], [0., 0.], [1., 1.], alpha=a, color='C0')
_ = sub.hist(mean_frac_dspectrum, 40, density=True, histtype='step', cumulative=True, color='k')
sub.set_xlabel(r'${\rm mean}_\lambda \langle (f_{\rm emu}  - f_{\rm fsps}) / f_{\rm fsps} \rangle$', fontsize=20)
sub.set_xlim(0., 0.03)
sub.set_ylabel('cumulative distribution', fontsize=20)
sub.set_ylim(0., 1.)
fig.savefig('fsps.%s.%svalid_emu.cum.png' % (model, ''.join(archs)), bbox_inches='tight') 
