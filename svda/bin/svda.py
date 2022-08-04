import os 
import glob 
import numpy as np 

from astropy.io import fits 
from astropy import table as aTable

from desitarget.targetmask import bgs_mask as main_bgs_mask
from desitarget.sv1.sv1_targetmask import bgs_mask as sv1_bgs_mask
from desitarget.sv3.sv3_targetmask import bgs_mask as sv3_bgs_mask


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
    coadd = aTable.Table.read(fcoadd, hdu=1)
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


def mwdust_transmission(ebv, band, photsys, match_legacy_surveys=False):
    """Convert SFD E(B-V) value to dust transmission 0-1 for band and photsys
    Args:
        ebv (float or array-like): SFD E(B-V) value(s)
        band (str): 'G', 'R', 'Z', 'W1', 'W2', 'W3', or 'W4'
        photsys (str or array of str): 'N' or 'S' imaging surveys photo system
    Returns:
        scalar or array (same as ebv input), Milky Way dust transmission 0-1
    If `photsys` is an array, `ebv` must also be array of same length.
    However, `ebv` can be an array with a str `photsys`.
    Also see `dust_transmission` which returns transmission vs input wavelength
    """
    if isinstance(photsys, str):
        r_band = extinction_total_to_selective_ratio(band, photsys, match_legacy_surveys=match_legacy_surveys)
        a_band = r_band * ebv
        transmission = 10**(-a_band / 2.5)
        return transmission
    else:
        photsys = np.asarray(photsys)
        if np.isscalar(ebv):
            raise ValueError('array photsys requires array ebv')
        if len(ebv) != len(photsys):
            raise ValueError('len(ebv) {} != len(photsys) {}'.format(
                len(ebv), len(photsys)))

        transmission = np.zeros(len(ebv))
        for p in np.unique(photsys):
            ii = (photsys == p)
            r_band = extinction_total_to_selective_ratio(band, p, match_legacy_surveys=match_legacy_surveys)
            a_band = r_band * ebv[ii]
            transmission[ii] = 10**(-a_band / 2.5)

        return transmission


def extinction_total_to_selective_ratio(band, photsys, match_legacy_surveys=False) :
    """Return the linear coefficient R_X = A(X)/E(B-V) where
    A(X) = -2.5*log10(transmission in X band),
    for band X in 'G','R' or 'Z' when
    photsys = 'N' or 'S' specifies the survey (BASS+MZLS or DECALS),
    or for band X in 'G', 'BP', 'RP' when photsys = 'G' (when gaia dr2)
    or for band X in 'W1', 'W2', 'W3', 'W4' when photsys is either 'N' or 'S'
    E(B-V) is interpreted as SFD.
    Args:
        band : 'G', 'R', 'Z', 'BP', 'RP', 'W1', 'W2', 'W3', or 'W4'
        photsys : 'N' or 'S'
    Returns:
        scalar, total extinction A(band) = -2.5*log10(transmission(band))
    """
    if match_legacy_surveys :
        # Based on the fit from the columns MW_TRANSMISSION_X and EBV
        # for the DR8 target catalogs and propagated in fibermaps
        # R_X = -2.5*log10(MW_TRANSMISSION_X) / EBV
        # It is the same value for the N and S surveys in DR8 and DR9 catalogs.
        R={"G_N":3.2140,
           "R_N":2.1650,
           "Z_N":1.2110,
           "G_S":3.2140,
           "R_S":2.1650,
           "Z_S":1.2110,
           "G_G":2.512,
           "BP_G":3.143,
           "RP_G":1.663,
        }
    else :
        # From https://desi.lbl.gov/trac/wiki/ImagingStandardBandpass
        # DECam u  3881.6   3.994
        # DECam g  4830.8   3.212
        # DECam r  6409.0   2.164
        # DECam i  7787.5   1.591
        # DECam z  9142.7   1.211
        # DECam Y  9854.5   1.063
        # BASS g  4772.1   3.258
        # BASS r  6383.6   2.176
        # MzLS z  9185.1   1.199
        # Consistent with the synthetic magnitudes and function dust_transmission

        R={"G_N":3.258,
           "R_N":2.176,
           "Z_N":1.199,
           "G_S":3.212,
           "R_S":2.164,
           "Z_S":1.211,
           "G_G":2.197,
           "BP_G":2.844,
           "RP_G":1.622,
        }

    # Add WISE from
    # https://github.com/dstndstn/tractor/blob/main/tractor/sfd.py#L23-L35
    R.update({
        'W1_N': 0.184,
        'W2_N': 0.113,
        'W3_N': 0.0241,
        'W4_N': 0.00910,
        'W1_S': 0.184,
        'W2_S': 0.113,
        'W3_S': 0.0241,
        'W4_S': 0.00910
        })

    assert(band.upper() in ["G","R","Z","BP","RP",'W1','W2','W3','W4'])
    assert(photsys.upper() in ["N","S","G"]), photsys.upper()
    return R["{}_{}".format(band.upper(),photsys.upper())]

