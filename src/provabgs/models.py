'''


module for sps models 


'''
import os 
import h5py 
import fsps
import pickle
import numpy as np 
from scipy.stats import sigmaclip
from scipy.special import gammainc
import scipy.interpolate as Interp
# --- astropy --- 
from astropy import units as U
from astropy.cosmology import Planck13 as cosmo
# --- gqp_mc --- 
from . import util as UT


class Model(object): 
    ''' base class object for SPS models 
    '''
    def __init__(self, **kwargs): 
        self._init_model(**kwargs)

    def sed(self, tt):
        ''' compute SED given a set of parameter values `tt`. 
        '''

    def _init_model(self, **kwargs) : 
        pass


class FSPS(Model): 
    '''
    '''
    def __init__(self): 
        super().__init__()

    def _init_model(self, model_name): 
        ''' initialize theta values 
        '''
        self.model_name = model_name # store model name 
        if self.model_name in ['emulator', 'fsps']: 
            names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh',
                    'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'tau']
        elif self.model_name == 'fsps_complexdust': 
            names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh',
                    'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2',
                    'dust_index']
        else: 
            raise NotImplementedError 
        self.theta_names = names 
        return None 

    def _ssp_initiate(self):
        ''' for models that use fsps, initiate fsps.StellarPopulation object 
        '''
        if 'fsps' not in self.model_name: 
            self._ssp = None
        elif self.model_name == 'fsps': 
            # initalize fsps object
            self._ssp = fsps.StellarPopulation(
                zcontinuous=1, # SSPs are interpolated to the value of logzsol before the spectra and magnitudes are computed
                sfh=0, # single SSP
                imf_type=1, # chabrier
                dust_type=2 # Calzetti (2000) 
                )
        elif self.model_name == 'fsps_complexdust': 
            self._ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=0,                  # sfh type 
                    dust_type=4,            # Kriek & Conroy attenuation curve. 
                    imf_type=1)             # chabrier 
        return None  


class DESIspeculator(Model): 
    ''' DESI speculator is a FSPS model emulator that is specifically trained
    for the DESI PROVABGS. It achieves <1% accuracy over the wavelength range
    2300 < lambda < 11000 Angstroms.

    more details to come 


    References
    ----------
    * Alsing et al.(2020) 
    '''
    def __init__(self, cosmo=cosmo): 
        super().__init__()
        
        # interpolators for speeding up cosmological calculations 
        _z = np.linspace(0.0, 0.5, 100)
        _tage = self.cosmo.age(_z).value
        _d_lum_cm = self.cosmo.luminosity_distance(_z).to(U.cm).value # luminosity distance in cm

        self._tage_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _tage, k=3)
        self._d_lum_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _d_lum_cm, k=3)

    def sed(self, tt, zred, wavelength=None, debug=False): 
        ''' compute the SED for a given set of parameter values and redshift.
       

        parameters
        ----------
        tt : np.ndarray
            array of parameter values. 
    
        zred : float, array 
            redshift 

        wavelength : np.ndarray 
            If specified, the model will interpolate the spectra to the specified 
            wavelengths.(default: None)  

        debug: bool
            If True, prints out a number lines to help debugging 


        returns
        -------
        tuple
            (outwave, outspec). outwave is an array of the output wavelenghts in
            Angstroms. outspec is an array of the redshifted output SED in units of 
            1e-17 * erg/s/cm^2/Angstrom
        '''
        tt      = np.atleast_2d(tt)
        zred    = np.atleast_1d(zred) 
        ntheta  = tt.shape[1]

        assert tt.shape[0] == zred.shape[0]
       
        outwave, outspec = [], [] 
        for _tt, _zred in zip(tt, zred): 
            # check redshift range
            assert (_zred >= 0.) and (_zred < 0.5), "outside of 0.0 <z < 0.5"

            # get tage
            tage = self._tage_z_interp(_zred)

            if debug: 
                print('Speculator.sed: redshift = %f' % zred)
                print('Speculator.sed: tage = %f' % tage) 

            # logmstar, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau, tage
            tt_arr = np.contatenate([_tt, [tage]])
            if debug: print('Speculator.sed: theta', tt_arr)
            
            # get SSP luminosity
            ssp_lum = self._emulator(tt_arr[1:]) 
            if debug: print('Speculator.sed: ssp lum', ssp_lum)

            # mass normalization
            lum_ssp = (10**tt_arr[0]) * ssp_lum

            # redshift the spectra
            w_z = self._emu_waves * (1. + _zred)
            d_lum = self._d_lum_z_interp(_zred) 
            flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + _zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang

            if wavelength is None: 
                outwave.append(w_z)
                outspec.append(flux_z)
            else: 
                outwave.append(wavelength)
                outspec.append(np.interp(outwave, w_z, flux_z, left=0, right=0))
        return np.array(outwave), np.array(outspec)

    def _emulator(self, tt):
        ''' forward pass through the the three speculator NN wave bins to
        predict SED 
        
        parameters
        ----------
        tt : np.ndarray
            array of parameter values [b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau, tage]
    
        returns
        -------
        array
            SSP flux in units of Lsun/A
        '''
        logflux = [] 
        for iwave in range(3): # wave bins

            # forward pass through the network
            act = []
            offset = np.log(np.sum(tt[0:4]))
            #layers = [(self._transform_theta(tt) - self._emu_theta_mean)/self._emu_theta_std]
            layers = [(tt - self._emu_theta_mean[iwave])/self._emu_theta_std[iwave]]

            for i in range(self._emu_n_layers[iwave]-1):
           
                # linear network operation
                act.append(np.dot(layers[-1], self._emu_W[iwave][i]) + self._emu_b[iwave][i])

                # pass through activation function
                layers.append((self._emu_beta[iwave][i] + (1.-self._emu_beta[iwave][i]) * 1./(1.+np.exp(-self._emu_alpha[iwave][i]*act[-1])))*act[-1])

            # final (linear) layer -> (normalized) PCA coefficients
            layers.append(np.dot(layers[-1], self._emu_W[iwave][-1]) + self._emu_b[iwave][-1])

            # rescale PCA coefficients, multiply out PCA basis -> normalized spectrum, shift and re-scale spectrum -> output spectrum
            _logflux = np.dot(layers[-1]*self._emu_pca_std[iwave] + self._emu_pca_mean[iwave], self._emu_pcas[iwave]) * self._emu_spec_std[iwave] + self._emu_spec_mean[iwave] + offset
            logflux.append(_logflux)

        return np.exp(np.concatenate(logflux)) 

    def _init_model(self): 
        ''' initialize the Speculator model 
        '''
        parameters = [
                'logmstar', 
                'beta1_sfh', 
                'beta2_sfh', 
                'beta3_sfh',
                'beta4_sfh', 
                'gamma1_zh', 
                'gamma2_zh', 
                'dust1', 
                'dust2',
                'dust_index']
        self._load_model_params()
        return None 

    def _load_model_params(self): 
        ''' read in pickle files that contains the parameters for the three
        wavelength bins of the DESI speculator model.  
        '''
        self._emu_W             = []
        self._emu_b             = []
        self._emu_alpha         = []
        self._emu_beta          = []
        self._emu_pcas          = []
        self._emu_pca_mean      = []
        self._emu_pca_std       = []
        self._emu_spec_mean     = []
        self._emu_spec_std      = []
        self._emu_theta_mean    = []
        self._emu_theta_std     = []
        self._emu_wave          = []

        self._emu_n_layers = []

        for i in range(3): # wavelength bins
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                'DESI.complexdust.Ntrain5e6.pca50.4x256.wave%i.pkl' % i), 'rb') 
            params = pickle.load(fpkl)
            fpkl.close()

            self._emu_W.append(params[0])
            self._emu_b.append(params[1])
            self._emu_alpha.append(params[2])
            self._emu_beta.append(params[3])
            self._emu_pcas.append(params[4])
            self._emu_pca_mean.append(params[5])
            self._emu_pca_std.append(params[6])
            self._emu_spec_mean.append(params[7])
            self._emu_spec_std.append(params[8])
            self._emu_theta_mean.append(params[9])
            self._emu_theta_std.append(params[10])
            self._emu_wave.append(params[11])

            self._emu_n_layers.append(len(params[0])) # number of network layers

        self._emu_waves = np.concatenate(self._emu_wave) 
        return None 
