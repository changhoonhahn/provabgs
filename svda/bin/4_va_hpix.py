'''


add columns to compiled healpix files:
    in particular compile z_max and M* (surviving stellar mass)  


'''
import os, sys
import h5py, glob
import numpy as np 

import svda as SVDA 

from astropy import units as U
from astropy.cosmology import Planck13
from scipy.interpolate import interp1d

from provabgs import util as UT
from provabgs import infer as Infer
from provabgs import models as Models

from speclite import filters as specFilter

#######################################################
# inputs
#######################################################
target = sys.argv[1]
survey = sys.argv[2]

####################################################
# gather healpixels  
####################################################
dat_dir = '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/'

hpixs = np.array([int(f.split('-')[-1].split('.')[0]) 
                  for f in glob.glob(os.path.join(dat_dir, f'provabgs-{survey}-bright-*.{target}.hdf5'))])
print('%i healpixels' % len(hpixs))

####################################################
m_nmf = Models.NMF(burst=True, emulator=True)

r_pass = specFilter.load_filters('decam2014-r')

def r_mag(w, f):
    ''' calculate r-band magnitude given w, f
    '''
    flux_z, w_z = r_pass.pad_spectrum(np.atleast_2d(f) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, w * U.Angstrom)
    maggies = r_pass.get_ab_maggies(flux_z, wavelength=w_z)
    return 22.5 - 2.5 * np.log10(maggies.as_array()[0][0] * 1e9)

def bgs_faint_color_cut(hpix): 
    ''' if True:    rfib < 20.75
        if False:   rfib < 21.5

    also return r-band fiber fraction 
    '''
    _x = SVDA.healpix(hpix, target=target, redux='fuji', survey=survey) 
    coadd = _x[0] 

    trans_g = SVDA.mwdust_transmission(coadd['EBV'], 'g',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)
    trans_r = SVDA.mwdust_transmission(coadd['EBV'], 'r',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)
    trans_z = SVDA.mwdust_transmission(coadd['EBV'], 'z',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)
    trans_w = SVDA.mwdust_transmission(coadd['EBV'], 'w1',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)

    g = 22.5 - 2.5*np.log10((coadd['FLUX_G'] / trans_g).clip(1e-16))
    r = 22.5 - 2.5*np.log10((coadd['FLUX_R'] / trans_r).clip(1e-16))
    z = 22.5 - 2.5*np.log10((coadd['FLUX_Z'] / trans_z).clip(1e-16))
    w1 = 22.5 - 2.5*np.log10((coadd['FLUX_W1'] / trans_w).clip(1e-16))

    schlegel_color = (z - w1) - 3/2.5 * (g - r) + 1.2

    return schlegel_color < 0., coadd['FIBERFLUX_R']/coadd['FLUX_R']


for hpix in hpixs: 
    with h5py.File(os.path.join(dat_dir, f'provabgs-{survey}-bright-{hpix}.{target}.hdf5'), 'r') as fhpix:
    
        if 'redshift' not in fhpix.keys(): 
            print()
            print('healpix %i is problematic' % hpix)
            print()
            continue
        
        zreds = fhpix['redshift'][...]
        if len(zreds) == 0: continue

        targids = fhpix['targetid'][...]
            
        # get posterior samples and log probabilities 
        _logp = fhpix['log_prob'][...]
        _logp = _logp.reshape((_logp.shape[0], _logp.shape[1] * _logp.shape[2]))
        
        _theta = fhpix['samples'][...]
        _theta = _theta.reshape((_theta.shape[0], _theta.shape[1] * _theta.shape[2], _theta.shape[3]))
        
        # get best-fit theta 
        theta_bfs = np.array([tt[imax,:] for imax, tt in zip(np.argmax(_logp, axis=1), _theta)])


    if target == 'BGS_FAINT': 
        faint_color_cut, f_fiber = bgs_faint_color_cut(hpix)
        
    # calculate z_max for each galaxy 
    zmax = np.zeros(len(zreds))
    for i, zred, theta_bf in zip(np.arange(len(zreds)), zreds, theta_bfs): 
        z_arr = np.linspace(zred, 0.6, 10)
        dlz = Planck13.luminosity_distance(z_arr).to(U.cm).value
        
        # get best-fit SED 
        w, f = m_nmf.sed(theta_bf[:-1], zred)
        
        w_z = w / (1. + zred) * (1 + z_arr[:,None])
        f_z = f * ((dlz[0]**2 / dlz**2) * (1 + zred)/(1 + z_arr))[:,None]
        
        r_arr = np.array([r_mag(_w, _f) for _w, _f in zip(w_z, f_z)])

        if target == 'BGS_BRIGHT': 

            if np.min(r_arr) > 19.5: 
                # already below the magnitude limit...
                zmax[i] = zred
            elif np.max(r_arr) < 19.5:
                # within the magnitude limit at 
                zmax[i] = 0.6
            else:
                fint_rz = interp1d(r_arr, z_arr, kind='cubic')
                zmax[i] = fint_rz(19.5)

        elif target == 'BGS_FAINT': 
            # scale by f_fiber from measurement (not inferred aperture 
            # correction because the target selection is done using fiber flux) 
            #f_z = f_z * theta_bf[-1] 
            r_fib = r_arr - 2.5 * np.log10(f_fiber[i]) # r-band fiber magnitude
            r_fib_cut = [21.5, 20.75][faint_color_cut[i]]

            if np.min(r_fib) > r_fib_cut: 
                # already below the fiber magnitude limit...
                zmax[i] = zred
            elif np.max(r_fib) < r_fib_cut:
                # within the fiber magnitude limit at 
                zmax[i] = 0.6
            else:
                fint_rz = interp1d(r_fib, z_arr, kind='cubic')
                zmax[i] = fint_rz(r_fib_cut)

    # calculate M* (surviving stellar mass) 
    tages = Planck13.age(zreds).value
    logmstar_bf = np.log10(m_nmf._surviving_mass(theta_bfs[:,:12], tages))

    logmstar = [] 
    for tage, tt in zip(tages, _theta): 
        logmstar.append(np.log10(m_nmf._surviving_mass(tt[:,:-1], tage)))
    logmstar = np.array(logmstar) 

    with h5py.File(os.path.join(dat_dir, f'provabgs-{survey}-bright-{hpix}.{target}.hdf5'), 'r+') as fhpix:
        if 'zmax' in fhpix.keys(): 
            fhpix['zmax'][...] = zmax
        else: 
            fhpix.create_dataset('zmax', data=zmax)
    
        if 'logMstar' in fhpix.keys(): 
            fhpix['logMstar'][...] = logmstar 
        else: 
            fhpix.create_dataset('logMstar', data=logmstar)

        if 'theta_bf' in fhpix.keys(): 
            fhpix['theta_bf'][...] = theta_bfs
        else: 
            fhpix.create_dataset('theta_bf', data=theta_bfs)
    
        if 'logMstar_bf' in fhpix.keys(): 
            fhpix['logMstar_bf'][...] = logmstar_bf 
        else: 
            fhpix.create_dataset('logMstar_bf', data=logmstar_bf)
