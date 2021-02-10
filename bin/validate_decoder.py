'''

validate the trained decoder

'''
import os, sys 
import numpy as np
from datetime import date
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib as mpl 
mpl.use('Agg') 
import matplotlib.pyplot as plt

#-------------------------------------------------------
# params 
#-------------------------------------------------------
name        = 'nmfburst'
nbatch      = 500
layers      = [1280, 1024, 512, 256, 64]
#-------------------------------------------------------

dat_dir = '/tigress/chhahn/provabgs/'
wave    = np.load(os.path.join(dat_dir, 'wave_fsps.npy')) 
nwave   = len(wave) 


class Decoder(nn.Module):
    def __init__(self, nfeat=1000, ncode=5, nhidden0=128, nhidden1=128, nhidden2=35, nhidden3=35, nhidden4=35, dropout=0.2):
        super(Decoder, self).__init__()

        self.ncode = int(ncode)

        self.dec0 = nn.Linear(ncode, nhidden4)
        self.d1 = nn.Dropout(p=dropout)
        self.dec1 = nn.Linear(nhidden4, nhidden3)
        self.d2 = nn.Dropout(p=dropout)
        self.dec2 = nn.Linear(nhidden3, nhidden2)
        self.d3 = nn.Dropout(p=dropout)
        self.dec3 = nn.Linear(nhidden2, nhidden1)
        self.d4 = nn.Dropout(p=dropout)
        self.dec4 = nn.Linear(nhidden1, nhidden0)
        self.d5 = nn.Dropout(p=dropout)
        self.outp = nn.Linear(nhidden0, nfeat)

    def decode(self, x):
        x = self.d1(F.leaky_relu(self.dec0(x)))
        x = self.d2(F.leaky_relu(self.dec1(x)))
        x = self.d3(F.leaky_relu(self.dec2(x)))
        x = self.d4(F.leaky_relu(self.dec3(x)))
        x = self.d5(F.leaky_relu(self.dec4(x)))
        x = self.outp(x)
        return x

    def forward(self, x):
        return self.decode(x)

    def loss(self, x, y):
        recon_y = self.forward(x)
        MSE = torch.sum(0.5 * (y - recon_y).pow(2))
        return MSE


# load in test data 
theta_test  = np.load(os.path.join(dat_dir, 'fsps.%s.theta.test.npy' % name))
lnspec_test = np.load(os.path.join(dat_dir, 'fsps.%s.lnspectrum.test.npy' % name)) 


lnspec_recon = [] 
for iw in range(3): # wave bins 
    wave_bin = [(wave < 4500), ((wave >= 4500) & (wave < 6500)), (wave >= 6500)][iw]

    # shift and scale of ln(spectra)
    shift_lnspec = np.load(os.path.join(dat_dir, 'fsps.%s.w%i.shift_lnspectrum.%i.npy' % (name, iw, nbatch)))
    scale_lnspec = np.load(os.path.join(dat_dir, 'fsps.%s.w%i.scale_lnspectrum.%i.npy' % (name, iw, nbatch)))

    lnspec_white_test = (lnspec_test[:,wave_bin] - shift_lnspec) / scale_lnspec

    n_test      = theta_test.shape[0] 
    n_theta     = theta_test.shape[1]
    n_lnspec    = len(shift_lnspec) 


    model = torch.load(os.path.join(dat_dir, 'decoder.fsps.%s.w%i.%ibatches.%s.pth' % (name, iw, nbatch, '_'.join([str(l) for l in layers]))))

    lnspec_white_recon_iw = model.forward(torch.tensor(theta_test, dtype=torch.float32))
    lnspec_recon_iw = scale_lnspec * lnspec_white_recon_iw.detach().numpy() + shift_lnspec
    lnspec_recon.append(lnspec_recon_iw) 

    print(lnspec_white_test[:5,:5])
    print(lnspec_white_recon_iw.detach().numpy()[:5,:5])

lnspec_recon = np.concatenate(lnspec_recon, axis=1)

print(lnspec_test[:5,:5])
print(lnspec_recon[:5,:5])

# plot a handful of SEDs
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
for ii, i in enumerate(np.random.choice(n_test, size=5, replace=False)):
    sub.plot(wave, np.exp(lnspec_recon[i]), c='C%i' % ii)
    sub.plot(wave, np.exp(lnspec_test[i]), c='C%i' % ii, ls='--')
sub.set_xlim(wave.min(), wave.max())

fig.savefig('fsps.%s.%i.%s.valid_decoder.sed.png' % (name, nbatch, str(date.today().isoformat())), bbox_inches='tight') 

# plot fractional reconstruction error 
frac_dspectrum = 1. - np.exp(lnspec_recon - lnspec_test)
frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
        [0.0005, 0.005, 0.025, 0.16, 0.5, 0.84, 0.975, 0.995, 0.9995], axis=0)

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.fill_between(wave, frac_dspectrum_quantiles[0],
        frac_dspectrum_quantiles[-1], fc='C0', ec='none', alpha=0.1, label='99.9%')
sub.fill_between(wave, frac_dspectrum_quantiles[1],
        frac_dspectrum_quantiles[-2], fc='C0', ec='none', alpha=0.2, label='99%')
sub.fill_between(wave, frac_dspectrum_quantiles[2],
        frac_dspectrum_quantiles[-3], fc='C0', ec='none', alpha=0.3, label='95%')
sub.fill_between(wave, frac_dspectrum_quantiles[3],
        frac_dspectrum_quantiles[-4], fc='C0', ec='none', alpha=0.5, label='68%')
sub.plot(wave, frac_dspectrum_quantiles[4], c='C0', ls='-') 
sub.plot(wave, np.zeros(len(wave)), c='k', ls=':') 

# mark +/- 1%
sub.plot(wave, 0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)
sub.plot(wave, -0.01 * np.ones(len(wave)), c='k', ls='--', lw=0.5)

sub.set_xlim(wave.min(), wave.max())
sub.set_ylabel(r'$(f_{\rm emu} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25) 
sub.set_ylim(-0.03, 0.03)
fig.savefig('fsps.%s.%i.%s.valid_decoder.png' % (name, nbatch, str(date.today().isoformat())), bbox_inches='tight') 
