'''

script to compile M* posteriors and calculate z_max for all galaxies
in a given healpix 


'''
import os, sys
import h5py
import glob
import numpy as np
from scipy.interpolate import interp1d

from astropy import units as U
from astropy.cosmology import Planck13

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
zred = f['redshift'][...]

theta_posteriors = mcmc.reshape((mcmc.shape[0], mcmc.shape[1] * mcmc.shape[2], mcmc.shape[3]))
print('%i galaxies' % theta_posteriors.shape[0])

# calculate z_max 
m_nmf = Models.NMF(burst=True, emulator=True)

r_pass = specFilter.load_filters('decam2014-r')

def r_mag(w, f):
    ''' calculate r-band magnitude given w, f 
    '''
    flux_z, w_z = r_pass.pad_spectrum(np.atleast_2d(f) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, w * U.Angstrom)
    maggies = r_pass.get_ab_maggies(flux_z, wavelength=w_z)
    return 22.5 - 2.5 * np.log10(maggies.as_array()[0][0] * 1e9)


zmaxes = np.zeros((theta_posteriors.shape[0], theta_posteriors.shape[1]))

for igal in range(theta_posteriors.shape[0]):
    z_arr = np.linspace(zred[igal], 0.6, 10)
    dlz = Planck13.luminosity_distance(z_arr).to(U.cm).value

    for j in range(theta_posteriors.shape[1]):
        theta = theta_posteriors[igal,j]
        r_arr = [] 
        
        w, f = m_nmf.sed(theta[:-1], zred[igal])
        
        for _z, _dlz in zip(z_arr, dlz): 
            w_z = w / (1. + zred[igal]) * (1 + _z)
            f_z = f * (dlz[0]**2 / _dlz**2) * (1 + zred[igal])/(1 + _z)
            
            r_arr.append(r_mag(w_z, f_z))        
            
        if np.min(r_arr) > 19.5: 
            zmaxes[igal,j] = zred[igal]
        elif np.max(r_arr) < 19.5: 
            zmaxes[igal,j] = 0.6
        else: 
            fint_rz = interp1d(r_arr, z_arr, kind='cubic')
            zmaxes[igal,j] = fint_rz(19.5)
        
fmstarz = h5py.File(
    os.path.join(dat_dir, 'provabgs-%s-%i.%s.mstar_zmax.hdf5' % (sample, hpix, target)), 'w')
fmstarz.create_dataset('logM', data=theta_posteriors[:,:,0]) 
fmstarz.create_dataset('zmax', data=zmaxes) 
fmstarz.close() 

