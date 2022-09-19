'''

script to interface with LOWZ data


'''
import os 
import glob 
import numpy as np 

from astropy.io import fits 
from astropy import table as aTable



def healpix(hpix, target='BGS_BRIGHT', redux='fuji', survey='sv3'):
    ''' read spectrophotometry and redshift of a petal from a cumulative tile
    for specified target  
    '''
    dir = os.path.join('/global/cfs/cdirs/desi/spectro/redux/', 
            redux, 'healpix', survey, 'bright', str(hpix)[:-2], str(hpix))
    
    # bgs target mask 
    if survey == 'sv3': 
        bgs_mask = sv3_bgs_mask
        key_targ = 'SV3_BGS_TARGET'
    elif survey == 'sv1': 
        bgs_mask = sv1_bgs_mask
        key_targ = 'SV1_BGS_TARGET'
    elif survey == 'main': 
        bgs_mask = main_bgs_mask
        key_targ = 'BGS_TARGET'

    # read spectra from coadd
    fcoadd = os.path.join(dir, 'coadd-%s-bright-%i.fits' % (survey, hpix))
    coadd = aTable.Table.read(fcoadd)
    frr = os.path.join(dir, 'redrock-%s-bright-%i.fits' % (survey, hpix))
    rr = aTable.Table.read(frr, hdu=1) 
        
    # select bgs targets with good fibers and good redshifts 
    is_bgs = (coadd[key_targ] & bgs_mask[target]) != 0 
    goodfiber = (coadd['COADD_FIBERSTATUS'] == 0)
    good_redshift = (
            (rr['ZWARN'] == 0) & 
            (rr['SPECTYPE'] == 'GALAXY') & 
            (rr['ZERR'] < 0.0005 * (1. + rr['Z'])) & 
            (rr['DELTACHI2'] > 40))
    zlim = (rr['Z'] > 0.) & (rr['Z'] < 0.6)
    
    is_target = (is_bgs & goodfiber & good_redshift & zlim) 
    
    spec = readDESIspec(fcoadd)
    # spectra  
    w_obs = np.concatenate([spec['wave_b'], spec['wave_r'], spec['wave_z']])
    f_obs = np.concatenate([spec['flux_b'], spec['flux_r'], spec['flux_z']],
            axis=1)
    i_obs = np.concatenate([spec['ivar_b'], spec['ivar_r'], spec['ivar_z']], 
            axis=1)

    # sort the wavelength
    isort = np.argsort(w_obs)
    w_obs = w_obs[isort]
    f_obs = f_obs[is_target,:][:,isort]
    i_obs = i_obs[is_target,:][:,isort]

    # photometry (de-redden)  
    coadd = coadd[is_target]
    trans_g = mwdust_transmission(coadd['EBV'], 'g',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)
    trans_r = mwdust_transmission(coadd['EBV'], 'r',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)
    trans_z = mwdust_transmission(coadd['EBV'], 'z',
            np.array(coadd['PHOTSYS']).astype(str),
            match_legacy_surveys=False)

    flux_g = coadd['FLUX_G'] / trans_g
    flux_r = coadd['FLUX_R'] / trans_r
    flux_z = coadd['FLUX_Z'] / trans_z
    fiberflux_r = coadd['FIBERFLUX_R'] / trans_r

    #list(coadd['FLUX_G', 'FLUX_R', 'FLUX_Z'].as_array())).copy()
    photo_flux = np.array([flux_g, flux_r, flux_z]).T 
    photo_ivar = np.array(list(coadd['FLUX_IVAR_G', 'FLUX_IVAR_R',
        'FLUX_IVAR_Z'].as_array())).copy()
    
    # fiber flux fraction estimate 
    f_fiber = coadd['FIBERFLUX_R'] / coadd['FLUX_R']
    sigma_f_fiber = f_fiber * coadd['FLUX_IVAR_R']**-0.5
    #assert np.isfinite(f_fiber)

    # redshift 
    zred = rr['Z'][is_target]

    return coadd, zred, photo_flux, photo_ivar, w_obs, f_obs, i_obs, f_fiber, sigma_f_fiber


