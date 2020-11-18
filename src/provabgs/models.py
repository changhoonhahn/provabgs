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
from astropy.cosmology import Planck13
# --- gqp_mc --- 
from . import util as UT


class Model(object): 
    ''' Base class object for different SPS models. The primary purpose of the
    `Model` class is to evaluate the SED given a set of parameter values. 
    '''
    def __init__(self, cosmo=None, **kwargs): 

        self._init_model(**kwargs)
        
        if cosmo is None: 
            self.cosmo = Planck13 # cosmology  

        # interpolators for speeding up cosmological calculations 
        _z = np.linspace(0.0, 0.5, 100)
        _tage = self.cosmo.age(_z).value
        _d_lum_cm = self.cosmo.luminosity_distance(_z).to(U.cm).value # luminosity distance in cm

        self._tage_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _tage, k=3)
        self._d_lum_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _d_lum_cm, k=3)

    def sed(self, tt):
        ''' compute SED given a set of parameter values `tt`. 
        '''

    def _init_model(self, **kwargs) : 
        pass


class FSPS(Model): 
    '''
    '''
    def __init__(self, name='default', cosmo=None): 
        self.name = name 

        super().__init__(cosmo=cosmo, name=name)

    def sed(self, tt, zred, wavelength=None, debug=False): 
        ''' compute the SED for a given set of parameter values and redshift.
       

        parameters
        ----------
        tt : array_like[Nsample, Nparam]
            parameter values. 
    
        zred : float or array_like[Nsample]
            redshift value(s)

        wavelength : array_like[Nwave] 
            If specified, the model will interpolate the spectra to the specified 
            wavelengths.
            (Default: None)  

        debug: bool
            If True, prints some debugging messages. 
            (Default: False) 


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
                print('FSPS.sed: redshift = %f' % zred)
                print('FSPS.sed: tage = %f' % tage) 

            tt_arr = np.concatenate([_tt, [tage]])
            if debug: print('FSPS.sed: theta', tt_arr)
            
            # get SSP luminosity
            ssp_lum = self._sps_model(tt_arr[1:])
            if debug: print('FSPS.sed: ssp lum', ssp_lum)

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

    def _fsps_nmf(self, tt): 
        ''' FSPS SPS model using NMF SFH and ZH bases with Kriek and Conroy
        attenuation curve. 


        Parameters 
        ----------
        tt : array_like[Nparam] 
            Parameter values for the **default setup**


        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 


        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A
        '''
        tt_sfh      = tt[:4] 
        tt_zh       = tt[4:6]
        tt_dust1    = tt[6]
        tt_dust2    = tt[7]
        tt_dust_index = tt[8]
        tage        = tt[9] 

        _t = np.linspace(0, tage, 50)
        tages   = max(_t) - _t + 1e-8 

        # Compute SFH and ZH
        sfh = np.sum(np.array([
            tt_sfh[i] *
            self._sfh_basis[i](_t)/np.trapz(self._sfh_basis[i](_t), _t) 
            for i in range(4)]), 
            axis=0)
        zh = np.sum(np.array([
            tt_zh[i] * self._zh_basis[i](_t) 
            for i in range(2)]), 
            axis=0)
        
        for i, tage, m, z in zip(range(len(tages)), tages, sfh, zh): 
            if m <= 0 and i != 0: # no star formation in this bin 
                continue
            self._ssp.params['logzsol'] = np.log10(z/0.0190) # log(Z/Zsun)
            self._ssp.params['dust1'] = tt_dust1
            self._ssp.params['dust2'] = tt_dust2 
            self._ssp.params['dust_index'] = tt_dust_index
            
            wave_rest, lum_i = self._ssp.get_spectrum(tage=tage, peraa=True) # in units of Lsun/AA
            # note that this spectrum is normalized such that the total formed
            # mass = 1 Msun

            if i == 0: lum_ssp = np.zeros(len(wave_rest))
            lum_ssp += m * lum_i 

        # the following normalization is to deal with the fact that
        # fsps.get_spectrum is normalized so that formed_mass = 1 Msun
        lum_ssp /= np.sum(sfh) 
        return wave_rest, lum_ssp

    def _init_model(self): 
        ''' initialize theta values 
        '''
        if self.name == 'default': 
            self.parameters = [
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
            self._sps_model = self._fsps_nmf
        else: 
            raise NotImplementedError 
        
        # initialize FSPS StellarPopulation object
        self._ssp_initiate() 
        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        if self.name == 'default': 
            self._ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=0,                  # sfh type 
                    dust_type=4,            # Kriek & Conroy attenuation curve. 
                    imf_type=1)             # chabrier 
        else: 
            raise NotImplementedError
        '''
            if self.model_name == 'vanilla': 
                ssp = fsps.StellarPopulation(
                        zcontinuous=1,          # interpolate metallicities
                        sfh=4,                  # sfh type 
                        dust_type=2,            # Calzetti et al. (2000) attenuation curve. 
                        imf_type=1)             # chabrier 
            elif self.model_name == 'vanilla_complexdust': 
                ssp = fsps.StellarPopulation(
                        zcontinuous=1,          # interpolate metallicities
                        sfh=4,                  # sfh type 
                        dust_type=4,            # Kriek & Conroy attenuation curve. 
                        imf_type=1)             # chabrier 
            elif self.model_name == 'vanilla_kroupa': 
                ssp = fsps.StellarPopulation(
                        zcontinuous=1,          # interpolate metallicities
                        sfh=4,                  # sfh type 
                        dust_type=2,            # Calzetti et al. (2000) attenuation curve. 
                        imf_type=2)             # chabrier 
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
        '''
        return None  


class DESIspeculator(Model): 
    ''' DESI speculator is a FSPS model emulator that is specifically trained
    for the DESI PROVABGS. It achieves <1% accuracy over the wavelength range
    2300 < lambda < 11000 Angstroms.

    more details to come 


    References
    ----------
    * Alsing et al.(2020) 


    Dev. Notes
    ----------
    * Nov 17, 2020: Tested `self._emulator` against DESI speculator training
      data and found good agreement. 
    * Nov 18, 2020: Tested `self.SFH`, `self.avgSFR`, `self.ZH`, `self.Z_MW`

    '''
    def __init__(self, cosmo=None): 
        super().__init__(cosmo=cosmo)
    
        # load NMF SFH and ZH bases 
        self._load_NMF_bases()

    def sed(self, tt, zred, wavelength=None, filters=None, debug=False): 
        ''' compute the redshifted spectral energy distribution (SED) for a given set of parameter values and redshift.
       

        Parameters
        ----------
        tt : array_like[Nsample,Nparam] 
            Parameter values that correspond to [log M*, b1SFH, b2SFH, b3SFH,
            b4SFH, g1ZH, g2ZH, 'dust1', 'dust2', 'dust_index']. 
    
        zred : float or array_like
            redshift of the SED 

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  
    
        filters : object
            Photometric bandpass filter to generate photometry.
            `speclite.FilterResponse` object. 

        debug: boolean
            If True, prints out a number lines to help debugging 


        Returns
        -------
        outwave : [Nsample, Nwave]
            output wavelengths in angstrom. 

        outspec : [Nsample, Nwave]
            the redshifted SED in units of 1e-17 * erg/s/cm^2/Angstrom.
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
            tt_arr = np.concatenate([_tt, [tage]])
            if debug: print('Speculator.sed: theta', tt_arr)
            
            # get SSP luminosity
            ssp_log_lum = self._emulator(tt_arr[1:]) 
            if debug: print('Speculator.sed: log(ssp lum)', ssp_log_lum)

            # mass normalization
            lum_ssp = np.exp(tt_arr[0] * np.log(10) + ssp_log_lum)

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

        if filters is None: 
            return np.array(outwave), np.array(outspec)
        else: 
            # calculate photometry from SEDs 
            maggies = filters.get_ab_maggies(
                    np.array(outspec) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom,
                    wavelength=np.array(outwave) * U.Angstrom) 

            return  np.array(outwave), np.array(outspec), maggies 

    def _emulator(self, tt):
        ''' forward pass through the the three speculator NN wave bins to
        predict SED 
        
        
        Parameters
        ----------
        tt : array_like[Nparam,]
            array of parameter values [b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau, tage]
    
        Returns
        -------
        logflux : array_like[Nwave,] 
            (natural) log of (SSP luminosity in units of Lsun/A)
        '''
        logflux = [] 
        for iwave in range(3): # wave bins

            # forward pass through the network
            act = []
            offset = np.log(np.sum(tt[:4]))
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

        return np.concatenate(logflux) 

    def SFH(self, tt, zred): 
        ''' SFH for a set of parameter values `tt` and redshift `zred`

        parameters
        ----------
        tt : array_like[Nparam,]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float
            redshift

        Returns 
        -------
        t : array_like[50,]
            age of the galaxy, linearly spaced from 0 to the age of the galaxy

        sfh : array_like[50,]
            star formation history at cosmic time t --- SFH(t) --- in units of
            Msun/**Gyr**. np.trapz(sfh, t) == Mstar 
        '''
        tt = np.atleast_2d(tt)
        tt_sfh = tt[:,1:5] # sfh basis coefficients 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0, tage, 50)
        
        # normalized basis out to t 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) 
            for i in range(4)])

        # caluclate normalized SFH
        sfh = np.sum(np.array([tt_sfh[:,i] * _basis[i][None,:] for i in range(4)]), axis=0)

        # multiply by stellar mass 
        sfh*= 10**tt[:,0]

        if tt.shape[0] == 1: 
            return t, sfh[0]
        else: 
            return t, sfh 
    
    def ZH(self, tt, zred):
        ''' metallicity history for a set of parameter values `tt` and redshift `zred`

        parameters
        ----------
        tt : array_like[Nparam,]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float
            redshift

        Returns 
        -------
        t : array_like[50,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[50,]
            metallicity at cosmic time t --- ZH(t) 
        '''
        tt = np.atleast_2d(tt)
        tt_zh = tt[:,5:7] # zh bases 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0, tage, 50)
    
        # metallicity basis is not normalized
        _z_basis = np.array([self._zh_basis[i](t) for i in range(2)]) 

        # get metallicity history
        zh = np.sum(np.array([tt_zh[:,i] * _z_basis[i][None,:] for i in range(2)]), axis=0) 
        if tt.shape[0] == 1: 
            return t, zh[0]
        else: 
            return t, zh 

    def avgSFR(self, tt, zred, dt=1.):
        ''' given a set of parameter values `tt` and redshift `zred`, calculate
        SFR averaged over `dt` Gyr. 

        parameters
        ----------
        tt : array_like[Ntheta, Nparam]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float or array_like[Ntheta] 
            redshift

        dt : float
            Gyrs to average the SFHs 
        '''
        tage = self.cosmo.age(zred).value # age in Gyr
        assert tage > dt 

        t, sfh = self.SFH(tt, zred) # get SFH 
        sfh = np.atleast_2d(sfh) 

        # add up the stellar mass formed during the dt time period 
        i_low = np.argmin(np.abs((t[-1] - t) - dt), axis=0)
        #np.clip(np.argmin(np.abs(t - (tage - dt)), axis=0), None, 48) 
        avsfr = np.trapz(sfh[:,i_low:], t[i_low:]) / (tage - t[i_low]) / 1e9
        return np.clip(avsfr, 0., None)
    
    def Z_MW(self, tt, zred):
        ''' given theta calculate mass weighted metallicity using the ZH NMF
        bases. 
        **NOT YET TESTED**
        '''
        tt = np.atleast_2d(tt) 
        t, sfh = self.SFH(tt, zred) # get SFH 
        _, zh = self.ZH(tt, zred) 
    
        # mass weighted average
        z_mw = np.trapz(zh * sfh, t) / (10**tt[:,0]) # np.trapz(sfh, t) should equal tt[:,0] 

        return np.clip(z_mw, 0, np.inf)

    def _init_model(self): 
        ''' initialize the Speculator model 
        '''
        self.parameters = [
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

        for i, npca in enumerate([50, 30, 30]): # wavelength bins
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                'DESI.complexdust.Ntrain5e6.pca%i.4x256.wave%i.pkl' % (npca, i)), 'rb') 
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

    def _load_NMF_bases(self): 
        ''' read in NMF SFH and ZH bases. These bases are used to reduce the
        dimensionality of the SFH and ZH. 
        '''
        fsfh = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'NMF_2basis_SFH_components_nowgt_lin_Nc4.txt')
        fzh = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'NMF_2basis_Z_components_nowgt_lin_Nc2.txt') 
        ft = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'sfh_t_int.txt') 

        nmf_sfh = np.loadtxt(fsfh) 
        nmf_zh  = np.loadtxt(fzh) 
        nmf_t   = np.loadtxt(ft) # look back time 

        self._nmf_t_lookback    = nmf_t
        self._nmf_sfh_basis     = nmf_sfh 
        self._nmf_zh_basis      = nmf_zh

        Ncomp_sfh = self._nmf_sfh_basis.shape[0]
        Ncomp_zh = self._nmf_zh_basis.shape[0]
    
        self._sfh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    max(self._nmf_t_lookback) - self._nmf_t_lookback, 
                    self._nmf_sfh_basis[i], k=1) 
                for i in range(Ncomp_sfh)
                ]
        self._zh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    max(self._nmf_t_lookback) - self._nmf_t_lookback, 
                    self._nmf_zh_basis[i], k=1) 
                for i in range(Ncomp_zh)]
        return None 
