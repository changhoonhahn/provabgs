'''

script to compile M* posteriors and calculate z_max for a given 
healpix 


'''
import os, sys
import h5py
import glob
import numpy as np
from scipy.interpolate import interp1d

from provabgs import util as UT
from provabgs import infer as Infer
from provabgs import models as Models

from speclite import filters as specFilter

sample  = sys.argv[1]
hpix    = int(sys.argv[2]) 
target  = sys.argv[3]

# read posterior samples for given healpix 
dat_dir = '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/'
fpost = os.path.join(dat_dir, 'provabgs-%s-%i.%s.hdf5' % (sample, hpix, target))

f = h5py.File(fpost, 'r')
mcmc = f['samples'][...][:,-100:,:,:]

theta_posteriors = mcmc.reshape((mcmc.shape[0], mcmc.shape[1] * mcmc.shape[2], mcmc.shape[3]))
print('%i galaxies' % theta_posteriors.shape[0])

# calculate z_max 
m_nmf = Models.NMF(burst=True, emulator=True)

r_pass = specFilter.load_filters('decam2014-r')

z_arr = np.linspace(0.01, 0.6, 10)
zmaxes = np.zeros((theta_posteriors.shape[0], theta_posteriors.shape[1]))

for igal in range(theta_posteriors.shape[0]):
    for j in range(theta_posteriors.shape[1]):
        theta = theta_posteriors[igal,j]
        r_arr = []
        for _z in z_arr:
            _, _, r_nmgy = m_nmf.sed(theta[:-1], _z, filters=r_pass)
            r_arr.append(22.5 - 2.5 * np.log10(r_nmgy[0]))
    
        if np.max(r_arr) > 19.5: 
            fint_rz = interp1d(r_arr, z_arr, kind='cubic')
            zmaxes[igal,j] = fint_rz(19.5)
        else: 
            zmaxes[igal,j] = 0.6 

fmstarz = h5py.File(
    os.path.join(dat_dir, 'provabgs-%s-%i.%s.mstar_zmax.hdf5' % (sample, hpix, target)), 'w')
fmstarz.create_dataset('logM', data=theta_posteriors[:,:,0]) 
fmstarz.create_dataset('zmax', data=zmaxes) 
fmstarz.close() 

