'''

script to estimate the probabilistic SMF by fitting a GMM 


'''
import os, sys 
import numpy as np
from tqdm.auto import trange
import astropy.table as aTable
from astropy.cosmology import Planck13

import copy
from nflows import transforms, distributions, flows

import torch
from torch import nn
from torch import optim
import torch.distributions as D

if torch.cuda.is_available():
    device = torch.device(type='cuda', index=0)
else:
    device = 'cpu'

#########################################################
# input
#########################################################
targ = sys.argv[1]
zmin = float(sys.argv[2])
zmax = float(sys.argv[3]) 
#########################################################

#########################################################
# read in BGS data
#########################################################
dat_dir = '/tigress/chhahn/provabgs/svda'
bgs = aTable.Table.read(os.path.join(dat_dir, 'BGS_ANY_full.provabgs.hdf5'))

has_posterior = (bgs['provabgs_z_max'].data != -999.)

if targ == 'bgs_bright': 
    is_bgs = bgs['is_bgs_bright']
elif targ == 'bgs_any': 
    is_bgs = (bgs['is_bgs_bright'] | bgs['is_bgs_faint'])

bgs = bgs[has_posterior & is_bgs]
print('%i %s galaxies with posteriors' % (len(bgs), targ))
#########################################################

class GaussianMixtureModel(nn.Module):
    def __init__(self, n_components: int=2):
        super().__init__()
        logweights = torch.zeros(n_components, )
        means   = torch.randn(n_components, ) + 10.
        logstdevs  = 0.1 * torch.tensor(np.random.randn(n_components, ))
        self.logweights = torch.nn.Parameter(logweights)
        self.means   = torch.nn.Parameter(means)
        self.logstdevs  = torch.nn.Parameter(logstdevs)

    def forward(self, x):
        mix  = D.Categorical(torch.exp(self.logweights))
        comp = D.Normal(self.means, torch.exp(self.logstdevs))
        gmm  = D.MixtureSameFamily(mix, comp)
        return - gmm.log_prob(x).mean()

    def log_prob(self, x):
        mix  = D.Categorical(torch.exp(self.logweights))
        comp = D.Normal(self.means, torch.exp(self.logstdevs))
        gmm  = D.MixtureSameFamily(mix, comp)
        return gmm.log_prob(x)

    def sample(self, N):
        mix  = D.Categorical(torch.exp(self.logweights))
        comp = D.Normal(self.means, torch.exp(self.logstdevs))
        gmm  = D.MixtureSameFamily(mix, comp)

        return gmm.sample(N)


def Loss(qphi, post, w):
    ''' calculate loss

    \sum_i^Ng w_i * \log \sum_j^Ns qphi(\theta_ij)

    '''
    logqphi = qphi.log_prob(post.flatten()[:,None]).reshape(post.shape)
    return -torch.sum(w * torch.logsumexp(logqphi, axis=1))

# redshift limit 
zlim = (bgs['Z_HP'].data > zmin) & (bgs['Z_HP'].data < zmax)

# calculate vmax
v_zmin = Planck13.comoving_volume(zmin).value * Planck13.h**3 # (Mpc/h)^3
v_zmax = Planck13.comoving_volume(zmax).value * Planck13.h**3 # (Mpc/h)^3
vmaxes = Planck13.comoving_volume(bgs['provabgs_z_max'].data).value * Planck13.h**3 # (Mpc/h)^3 

# calculate weights 
w_import = (v_zmax - v_zmin) / (vmaxes.clip(v_zmin, v_zmax) - v_zmin) 
w_import *= bgs['provabgs_w_zfail'].data * bgs['provabgs_w_fibassign']

x_data = torch.tensor(bgs['provabgs_logMstar'].data[zlim].astype(np.float32)).to(device)
w_data = torch.tensor(w_import[zlim].astype(np.float32)).to(device)

batch_size = 128
Ntrain = int(0.9 * x_data.shape[0])
Nvalid = x_data.shape[0] - Ntrain 

trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_data[:Ntrain], w_data[:Ntrain]),
        batch_size=batch_size,
        shuffle=True)

validloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_data[Ntrain:], w_data[Ntrain:]),
        batch_size=batch_size)

lr          = 1e-3
num_iter    = 1000
patience    = 20
n_model     = 5 

best_flows, best_valid_losses, vls = [], [], []
for i in range(n_model): 
    ncomp = int(np.ceil(np.exp(np.random.uniform(np.log(5), np.log(100)))))
    flow = GaussianMixtureModel(n_components=ncomp)
    flow.to(device)
    print('GMM with %i components' % ncomp)

    # parameters = [weights, means, stdevs]
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=num_iter)

    best_epoch, best_valid_loss = 0, np.inf
    valid_losses = []

    t = trange(num_iter, leave=False)
    for epoch in t:
        train_loss = 0.
        for batch in trainloader: 
            optimizer.zero_grad()
            _post, _w = batch
            _post = _post.to(device)
            _w = _w.to(device)

            loss = Loss(flow, _post, _w)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(trainloader.dataset)

        with torch.no_grad():
            valid_loss = 0.
            for batch in validloader: 
                _post, _w = batch
                _post = _post.to(device)
                _w = _w.to(device)

                loss = Loss(flow, _post, _w)                
                valid_loss += loss.item()
            valid_loss /= len(validloader.dataset)           
            valid_losses.append(valid_loss)

        scheduler.step()

        t.set_description('Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % 
                          (epoch, train_loss, valid_loss), refresh=False)

        if valid_loss < best_valid_loss: 
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_flow = copy.deepcopy(flow)
        else: 
            if best_epoch < epoch - patience: 
                print('>>>%i \t %.5e' % (epoch, best_valid_loss))
                break

    torch.save(best_flow, os.path.join(dat_dir, 'psmf.gmm.%s.z%.1f_%.1f.%i.pt' % (targ, zmin, zmax, i)))
    best_flows.append(best_flow)
    best_valid_losses.append(best_valid_loss)
    vls.append(valid_losses)
    
ibest = np.argmin(best_valid_losses)
torch.save(best_flows[ibest], os.path.join(dat_dir, 'psmf.gmm.%s.z%.1f_%.1f.best.pt' % (targ, zmin, zmax)))
