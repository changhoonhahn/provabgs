import os 
import glob 
import numpy as np 

from astropy.io import fits 
from astropy import table as aTable

from desitarget.targetmask import bgs_mask as main_bgs_mask
from desitarget.sv1.sv1_targetmask import bgs_mask as sv1_bgs_mask
from desitarget.sv3.sv3_targetmask import bgs_mask as sv3_bgs_mask


def cumulative_tile_petal(tileid, i_petal, target='BGS_BRIGHT', redux='fuji', survey='sv3'):
    ''' read spectrophotometry and redshift of a petal from a cumulative tile
    for specified target  
    '''
    dir = os.path.join('/global/cfs/cdirs/desi/spectro/redux/', redux)

    subdirs = glob.glob(os.path.join(dir, 'tiles', 'cumulative', str(tileid), '*'))

    assert len(subdirs) == 1
    
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
    fcoadd = glob.glob(os.path.join(subdirs[0], 'coadd-%i-%i-*.fits' % (i_petal, tileid)))[0]
    coadd = aTable.Table.read(fcoadd)
    frr = glob.glob(os.path.join(subdirs[0], 'redrock-%i-%i-*.fits' % (i_petal, tileid)))[0]
    rr = aTable.Table.read(frr, hdu=1) 
        
    # select bgs targets with good fibers and good redshifts 
    is_bgs = (coadd[key_targ] & bgs_mask[target]) != 0 
    goodfiber = (coadd['COADD_FIBERSTATUS'] == 0)
    good_redshift = (
            (rr['ZWARN'] == 0) & 
            (rr['SPECTYPE'] == 'GALAXY') & 
            (rr['ZERR'] < 0.0005 * (1. + rr['Z'])) & 
            (rr['DELTACHI2'] > 40))

    is_target = (is_bgs & goodfiber & good_redshift) 
    
    spec = readDESIspec(fcoadd)
    # spectra  
    w_obs = np.concatenate([spec['wave_b'], spec['wave_r'], spec['wave_z']])
    f_obs = np.concatenate([spec['flux_b'], spec['flux_r'], spec['flux_z']],
            axis=1)
    i_obs = np.concatenate([spec['ivar_b'], spec['ivar_r'], spec['ivar_z']], 
            axis=1)

    # photometry 
    photo_flux = np.array(list(coadd['FLUX_G', 'FLUX_R',
        'FLUX_Z'].as_array())).copy()
    photo_ivar = np.array(list(coadd['FLUX_IVAR_G', 'FLUX_IVAR_R',
        'FLUX_IVAR_Z'].as_array())).copy()
    
    # fiber flux fraction estimate 
    f_fiber = (coadd['FIBERFLUX_R'] / coadd['FLUX_R'])
    sigma_f_fiber = f_fiber * coadd['FLUX_IVAR_R']**-0.5
    #assert np.isfinite(f_fiber)

    # redshift 
    zred = rr['Z'] 

    # sort the wavelength
    isort = np.argsort(w_obs)

    return coadd[is_target], zred[is_target], photo_flux[is_target], photo_ivar[is_target], w_obs[isort], f_obs[is_target,:][:,isort], i_obs[is_target,:][:,isort], f_fiber[is_target], sigma_f_fiber[is_target]


def readDESIspec(ffits): 
    ''' read DESI spectra fits file

    :params ffits: 
        name of fits file  
    
    :returns spec:
        dictionary of spectra
    '''
    fitobj = fits.open(ffits)
    
    spec = {} 
    for i_k, k in enumerate(['wave', 'flux', 'ivar']): 
        spec[k+'_b'] = fitobj[3+i_k].data
        spec[k+'_r'] = fitobj[8+i_k].data
        spec[k+'_z'] = fitobj[13+i_k].data

    spec['TARGETID'] = fitobj[1].data['TARGETID']
    return spec 

