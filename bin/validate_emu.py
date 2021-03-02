'''

validate the PCA NN 

'''
import os, sys 
import pickle
import numpy as np

from speculator import SpectrumPCA
from speculator import Speculator

import matplotlib as mpl 
mpl.use('Agg') 
import matplotlib.pyplot as plt

#-------------------------------------------------------
# input 
#-------------------------------------------------------
#model = 'nmfburst'
#n_pcas  = [30, 60, 30, 30, 30, 30] # number of PCA components 
#archs = ['16x256', '16x256', '16x256', '16x256', '16x256', '16x256'] # architectures

model = 'nmf_bases'
n_pcas = [50, 30, 30]
archs = ['8x256', '8x256', '8x256'] # architectures
#n_pcas = [30, 30, 30, 30, 30, 30]
#archs = ['10x256', '10x256', '10x256', '10x256', '10x256', '10x256'] # architectures

#model = 'burst'
#n_pcas = [50, 30, 30]
#archs = ['5x256', '5x256', '5x256'] # architectures

nbatch = 500 
desc = 'nbatch250'
#dat_dir = '/Users/chahah/data/provabgs/'
dat_dir = '/tigress/chhahn/provabgs/'

# read in wavelenght values 
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy'))

n_wavebin = len(n_pcas)
if n_wavebin == 6: 
    wave_bins = [ 
            (wave < 3000),
            (wave >= 3000) & (wave < 4000),
            (wave >= 4000) & (wave < 5000),
            (wave >= 5000) & (wave < 6000), 
            (wave >= 6000) & (wave < 7000),
            (wave >= 7000)]
elif n_wavebin == 3: 
    wave_bins = [
            (wave < 4500), 
            (wave >= 4500) & (wave < 6500),
            (wave >= 6500)]


# plot loss 
losses = [
        np.loadtxt(os.path.join(dat_dir,
            'fsps.%s.seed0_%i.%iw%i.pca%i.%s.%s.loss.dat' % (model, nbatch-1, n_wavebin, i, n_pcas[i], archs[i], desc)))
            for i in range(len(wave_bins))
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
fig.savefig('fsps.%s.%s.%s.%s.valid_emu.loss.png' % (model, '_'.join(str(nn) for nn in n_pcas), '.'.join(archs), desc), bbox_inches='tight') 


# read in test parameters and data
theta_test      = np.load(os.path.join(dat_dir,
    'fsps.%s.theta.test.npy' % model)).astype(np.float32)
_lnspec_test    = np.load(os.path.join(dat_dir, 
    'fsps.%s.lnspectrum.test.npy' % model))

lnspec_test = []
for wave_bin in wave_bins: 
    lnspec_test.append(_lnspec_test[:,wave_bin])
lnspec_test = np.concatenate(lnspec_test, axis=1) 

_wave = [] 
for wave_bin in wave_bins: 
    _wave.append(wave[wave_bin]) 
_wave = np.concatenate(_wave)

# load speculator 
emus = [
        Speculator(
            restore=True, 
            restore_filename=os.path.join(dat_dir, 
                'fsps.%s.seed0_%i.%iw%i.pca%i.%s.%s' % 
                (model, nbatch-1, n_wavebin, i, n_pcas[i], archs[i], desc))
            )
        for i in range(len(wave_bins))
        ]

def emu_lnspec(theta):
    lnspec = [emus[i].log_spectrum(theta) for i in range(len(emus))]
    return np.concatenate(lnspec, axis=1)

# reconstructed ln(spec) 
lnspec_recon = emu_lnspec(theta_test) 

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
fig.savefig('fsps.%s.%s.%s.%s.valid_emu.png' % (model, '_'.join(str(nn) for nn in n_pcas), '.'.join(archs), desc), bbox_inches='tight') 


# plot cumulative fractional error
mean_frac_dspectrum = np.mean(np.abs(frac_dspectrum), axis=1)
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
fig.savefig('fsps.%s.%s.%s.%s.valid_emu.cum.png' % (model, '_'.join(str(nn) for nn in n_pcas), '.'.join(archs), desc), bbox_inches='tight') 

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
fig.savefig('fsps.%s.%s.%s.%s.outlier.png' % (model, '_'.join(str(nn) for nn in n_pcas), '.'.join(archs), desc), bbox_inches='tight') 
