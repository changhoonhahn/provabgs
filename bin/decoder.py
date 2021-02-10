'''

python script to train a decoder 


'''
import os, sys 
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

#-------------------------------------------------------
# params 
#-------------------------------------------------------
name = sys.argv[1]
i_wave = int(sys.argv[2])
nbatch = int(sys.argv[3]) 
#-------------------------------------------------------
nhidden0 = 1280 
nhidden1 = 1024
nhidden2 = 512
nhidden3 = 256 
nhidden4 = 64 

# load in training set  
dat_dir = '/tigress/chhahn/provabgs/'
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy'))
wave_bin = [(wave < 4500), ((wave >= 4500) & (wave < 6500)), (wave >= 6500)][i_wave]
nwave = len(wave[wave_bin])

# average and sigma of ln(spectra)
fshift = os.path.join(dat_dir, 'fsps.%s.w%i.shift_lnspectrum.%i.npy' % (name, i_wave, nbatch))
if not os.path.isfile(fshift): 
    shift_lnspec = np.zeros(nwave)
    for i in range(nbatch): 
        shift_lnspec += np.mean(np.load(os.path.join(dat_dir, 'fsps.%s.lnspectrum.seed%i.npy' % (name, i))), axis=0)[wave_bin]/float(nbatch)
    np.save(fshift, shift_lnspec)
else: 
    shift_lnspec = np.load(fshift) 

fscale = os.path.join(dat_dir, 'fsps.%s.w%i.scale_lnspectrum.%i.npy' % (name, i_wave, nbatch))
if not os.path.isfile(fscale): 
    scale_lnspec = np.zeros(nwave) 
    for i in range(nbatch): 
        scale_lnspec += np.std(np.load(os.path.join(dat_dir, 'fsps.%s.lnspectrum.seed%i.npy' % (name, i))), axis=0)[wave_bin]/float(nbatch)
    np.save(fscale, scale_lnspec)
else: 
    scale_lnspec = np.load(fscale) 

print(shift_lnspec)
print(scale_lnspec) 

n_theta     = (np.load(os.path.join(dat_dir, 'fsps.%s.theta.seed0.npy' % name))).shape[1]
n_lnspec    = len(shift_lnspec) 
print('n_theta = %i' % n_theta) 
print('n_lnspec = %i' % n_lnspec) 


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


def train(batch_size): #model, optimizer, epoch, min_valid_loss, badepochs
    ''' train by looping through files with 10,000 SEDs and for each file, looping through batch_size batches.
    '''
    model.train()
    train_loss = 0
    for i in range(nbatch): #batch_idx, data in enumerate(train_loader): 
        tt = torch.tensor(np.load(os.path.join(dat_dir,
            'fsps.%s.theta.seed%i.npy' % (name, i))), 
            dtype=torch.float32)
        lns = torch.tensor((np.load(os.path.join(dat_dir, 
            'fsps.%s.lnspectrum.seed%i.npy' % (name, i)))[:,wave_bin] -
            shift_lnspec)/scale_lnspec, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tt, lns),
                batch_size=batch_size)
        _train_loss = 0
        for data in train_loader: 
            _tt, _lns = data
            optimizer.zero_grad()
            loss = model.loss(_tt, _lns)
            loss.backward()
            _train_loss += loss.item()
            optimizer.step()
        _train_loss /= len(train_loader.dataset)
        train_loss += _train_loss 
    train_loss /= nbatch
    return train_loss


class EarlyStopper:
    def __init__(self, precision=1e-3, patience=10):
        self.precision = precision
        self.patience = patience
        self.badepochs = 0
        self.min_valid_loss = float('inf')

    def step(self, valid_loss):
        if valid_loss < self.min_valid_loss*(1-self.precision):
            self.badepochs = 0
            self.min_valid_loss = valid_loss
        else:
            self.badepochs += 1
        return not (self.badepochs == self.patience)


# output loss
_floss = os.path.join(dat_dir, 
    'decoder.fsps.%s.w%i.%ibatches.%i_%i_%i_%i_%i.loss' %
    (name, i_wave, nbatch, nhidden0, nhidden1,  nhidden2, nhidden3, nhidden4))
floss = open(_floss, 'w')
floss.close()

epochs = 200
n_config = 1
batch_sizes = [100, 1000, 5000, 10000]
ibatch = 0 

for config in range(n_config):
    dropout = 0. #0.9*np.random.uniform()

    print('config %i, dropout = %0.2f; 5 hidden layers with %i, %i, %i, %i, %i nodes' % (config, dropout, nhidden0, nhidden1, nhidden2, nhidden3, nhidden4))
    model = Decoder(nfeat=n_lnspec, nhidden0=nhidden0, nhidden1=nhidden1, nhidden2=nhidden2, nhidden3=nhidden3, nhidden4=nhidden4, ncode=n_theta, dropout=dropout)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    stopper = EarlyStopper(patience=10)
    
    for batch_size in batch_sizes: 
        for epoch in range(1, epochs + 1):
            train_loss = train(batch_size)
            print('====> Epoch: %i BATCH SIZE %i TRAINING Loss: %.2e' % (epoch, batch_size, train_loss))
            floss = open(_floss, "a") # append
            floss.write('%i \t %i \t %.5e \n' % (batch_size, epoch, train_loss))
            floss.close()

            scheduler.step(train_loss)
            if (not stopper.step(train_loss)) or (epoch == epochs):
                print('Stopping')
                print('====> Epoch: %i BATCH SIZE %i TRAINING Loss: %.2e' % (epoch, batch_size, train_loss))
                torch.save(model, os.path.join(dat_dir, 'decoder.fsps.%s.w%i.%ibatches.%i_%i_%i_%i_%i.final.pth' % (name, i_wave, nbatch, nhidden0, nhidden1,  nhidden2, nhidden3, nhidden4)))
                break 
            torch.save(model, os.path.join(dat_dir, 'decoder.fsps.%s.w%i.%ibatches.%i_%i_%i_%i_%i.pth' % (name, i_wave, nbatch, nhidden0, nhidden1,  nhidden2, nhidden3, nhidden4)))
