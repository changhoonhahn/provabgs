import os 
import numpy as np 
from astropy import table as aTable

from provabgs import util as UT 

def get_spectrophotometry(igal, sample='sv_everst_faint_sf.fits'):
    ''' get spectrophotometry by getting the spectra file
    '''
    dat_dir = '/global/cfs/cdirs/desi/users/raga19/sed_fitting_inputs/'
    gals = aTable.Table.read(os.path.join(dat_dir, sample))

    # get redshift
    zred_i = gals['Z'][igal]

    # get photometry
    spec_table = aTable.Table.read(gals['Spectra_Path'][igal])

    spec_i = UT.readDESIspec(gals['Spectra_Path'][igal])

    istarg = (spec_i['TARGETID'] == gals['TARGETID'][igal])
    assert np.sum(istarg) > 0
    iigal = np.arange(len(spec_i['TARGETID']))[istarg]

    # get photometry
    photo_flux_i = np.array(list(spec_table['FLUX_G', 'FLUX_R', 'FLUX_Z'].as_array()[iigal][0]))
    photo_ivar_i = np.array(list(spec_table['FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z'].as_array()[iigal][0]))
    # get spectra
    w_obs = np.concatenate([spec_i['wave_b'], spec_i['wave_r'], spec_i['wave_z']])
    f_obs = np.concatenate([spec_i['flux_b'][iigal][0], spec_i['flux_r'][iigal][0], spec_i['flux_z'][iigal][0]])
    i_obs = np.concatenate([spec_i['ivar_b'][iigal][0], spec_i['ivar_r'][iigal][0], spec_i['ivar_z'][iigal][0]])

    f_fiber = (spec_table['FIBERFLUX_R'] / spec_table['FLUX_R'])[iigal]
    sigma_f_fiber = f_fiber * spec_table['FLUX_IVAR_R'][iigal]**-0.5
    assert np.isfinite(f_fiber)

    isort = np.argsort(w_obs)

    return zred_i, photo_flux_i, photo_ivar_i, w_obs[isort], f_obs[isort], i_obs[isort], f_fiber, sigma_f_fiber
