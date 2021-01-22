'''

validate the trained decoder

'''
import os 
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

#-------------------------------------------------------
# params 
#-------------------------------------------------------
model = sys.argv[1]
nbatch = int(sys.argv[2]) 
#-------------------------------------------------------

dat_dir = '/tigress/chhahn/provabgs/'
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy')) 

# load in test data 
theta_test  = np.load(os.path.join(dat_dir, 'fsps.%s.theta.test.npy' % model))
lnspec_test = np.load(os.path.join(dat_dir, 'fsps.%s.lnspectrum.test.npy' % model)) 

# average and sigma of ln(spectra)
mu_lnspec = np.zeros(nwave)
for i in range(nbatch): 
    mu_lnspec += np.mean(np.load(os.path.join(dat_dir, 'fsps.%s.lnspectrum.seed%i.npy' % (name, i))), axis=0)/float(nbatch)

sig_lnspec = np.zeros(nwave) 
for i in range(nbatch): 
    sig_lnspec += np.sum((np.load(os.path.join(dat_dir, 'fsps.%s.lnspectrum.seed%i.npy' % (name, i))) - mu_lnspec)**2, axis=0)/float(nbatch)
sig_lnspec = np.sqrt(sig_lnspec) 

n_test      = theta_test.shape[0] 
n_theta     = theta_test.shape[1]
n_lnspec    = len(mu_lnspec) 

lnspec_test_white = (lnspec_test - mu_lnspec) / sig_lnspec

class Decoder(nn.Module):
    def __init__(self, nfeat=1000, ncode=5, nhidden=128, nhidden2=35, dropout=0.2):
        super(Decoder, self).__init__()

        self.ncode = int(ncode)

        self.decd = nn.Linear(ncode, nhidden2)
        self.d3 = nn.Dropout(p=dropout)
        self.dec2 = nn.Linear(nhidden2, nhidden)
        self.d4 = nn.Dropout(p=dropout)
        self.outp = nn.Linear(nhidden, nfeat)

    def decode(self, x):
        x = self.d3(F.leaky_relu(self.decd(x)))
        x = self.d4(F.leaky_relu(self.dec2(x)))
        x = self.outp(x)
        return x

    def forward(self, x):
        return self.decode(x)

    def loss(self, x, y):
        recon_y = self.forward(x)
        MSE = torch.sum(0.5 * (y - recon_y).pow(2))
        return MSE

model = torch.load(os.path.join(dat_dir, 'decoder.fsps.%s.%ibatches.pth' % (name, nbatch)))

lnspec_white_recon = model.forward(torch.tensor(theta_test, dtype=torch.float32))
lnspec_recon = sig_lnspec * (lnspec_white_recon.detach().numpy() + mu_lnspec)

# plot a handful of SEDs
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
for i in np.random.choice(n_test, size=5, replace=False):
    sub.plot(wave, np.exp(lnspec_recon[i]), c='C%i' % i)
    sub.plot(wave, np.exp(lnspec_test[i]), c='C%i' % i, ls='--')
sub.set_xlim(wave.min(), wave.max())
fig.savefig('fsps.%s.%i.valid_decoder.sed.png' % (model, n_test), bbox_inches='tight') 

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
sub.set_ylim(-0.1, 0.1)
fig.savefig('fsps.%s.%i.valid_decoder.sed.png' % (model, n_test), bbox_inches='tight') 
