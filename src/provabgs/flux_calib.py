'''

module for spectrophotometric flux calibration 

'''
import numpy as np 


def no_flux_factor(tt, flux): 
    ''' flux calibration is a constant factor across all wavelenghts. This is
    the simplest flux calibration model we have. 

    Parameter
    ---------
    tt : None

    flux : array_like[Nwave,]
        SED that flux calibration is being applied to. 
    '''
    if isinstance(flux, list): 
        return np.concatenate(flux) 
    return flux 


def constant_flux_factor(tt, flux): 
    ''' flux calibration is a constant factor across all wavelenghts. This is
    the simplest flux calibration model we have. 

    Parameter
    ---------
    tt : array_like[1,]
         flux calibration factor 

    flux : array_like[Nwave,]
        SED that flux calibration is being applied to. 
    '''
    return tt * flux 


def constant_flux_DESI_arms(tt, flux_list):
    ''' flux calibration is a constant factor for each DESI spectrograph arm 
    '''
    tt_b, tt_r, tt_z = tt 
    flux_b, flux_r, flux_z = flux_list 
    return np.concatenate([tt_b * flux_b, tt_r * flux_r, tt_z * flux_z])
