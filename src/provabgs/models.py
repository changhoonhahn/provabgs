'''


module for sps models 


'''
import os 
import h5py 
import pickle
import warnings
import numpy as np 
from scipy.stats import sigmaclip
from scipy.special import gammainc
import scipy.interpolate as Interp
# --- astropy --- 
from astropy import units as U
from astropy.cosmology import Planck13
# --- gqp_mc --- 
from . import util as UT

try: 
    import fsps
except ImportError:
    warnings.warn('import error with fsps; only use emulators')


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
        self._z_tage_interp = \
                Interp.InterpolatedUnivariateSpline(_tage[::-1], _z[::-1], k=3)
        self._d_lum_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _d_lum_cm, k=3)
    
    def sed(self, tt, zred, vdisp=0., wavelength=None, resolution=None,
            filters=None, debug=False):
        ''' compute the redshifted spectral energy distribution (SED) for a
        given set of parameter values and redshift.
       

        Parameters
        ----------
        tt : 2-d array
            [Nsample,Nparam] SPS parameters     

        zred : float or array_like
            redshift of the SED 

        vdisp : float or array_like
            velocity dispersion  

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  

        resolution : array_like[N,Nwave]
            resolution matrix (e.g. DESI data provides a resolution matrix)  
    
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
        vdisp   = np.atleast_1d(vdisp) 
        ntheta  = tt.shape[1]

        assert tt.shape[0] == zred.shape[0]
       
        outwave, outspec, maggies = [], [], [] 
        for _tt, _zred in zip(tt, zred): 

            if debug: print('Model.sed: redshift = %f' % _zred)
            _tage = self.cosmo.age(_zred).value 

            # get SSP luminosity
            wave_rest, lum_ssp = self._sps_model(_tt, _tage)
            if debug: print('Model.sed: ssp lum', lum_ssp)

            # redshift the spectra
            w_z = wave_rest * (1. + _zred)
            d_lum = self._d_lum_z_interp(_zred) 
            flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + _zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang

            # apply velocity dispersion 
            wave_smooth, flux_smooth = self._apply_vdisp(w_z, flux_z, vdisp)
            
            if wavelength is None: 
                outwave.append(wave_smooth)
                outspec.append(flux_smooth)
            else: 
                outwave.append(wavelength)

                # resample flux to input wavelength  
                resampflux = UT.trapz_rebin(wave_smooth, flux_smooth, xnew=wavelength) 

                if resolution is not None: 
                    # apply resolution matrix 
                    _i = 0 
                    for res in np.atleast_1d(resolution):
                        _res = UT.Resolution(res) 
                        resampflux[_i:_i+res.shape[-1]] = _res.dot(resampflux[_i:_i+res.shape[-1]]) 
                        _i += res.shape[-1]
                outspec.append(resampflux) 

            if filters is not None: 
                # calculate photometry from SEDs 
                _maggies = filters.get_ab_maggies(np.atleast_2d(flux_z) *
                        1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_z *
                        U.Angstrom)
                maggies.append(np.array(list(_maggies[0])) * 1e9)

        if len(outwave) == 1: 
            outwave = outwave[0] 
            outspec = outspec[0] 
            if filters is not None: maggies = maggies[0]
        else: 
            outwave = np.array(outwave)
            outspec = np.array(outspec) 
            if filters is not None: maggies = np.array(maggies)

        if filters is None: 
            return outwave, outspec
        else: 
            return outwave, outspec, maggies
    
    def _init_model(self, **kwargs) : 
        return None 

    def _apply_vdisp(self, wave, flux, vdisp): 
        ''' apply velocity dispersion by first rebinning to log-scale
        wavelength then convolving vdisp. 

        Notes
        -----
        * code lift from https://github.com/desihub/desigal/blob/d67a4350bc38ae42cf18b2db741daa1a32511f8d/py/desigal/nyxgalaxy.py#L773
        * confirmed that it reproduces the velocity dispersion calculations in
        prospector
        (https://github.com/bd-j/prospector/blob/41cbdb7e6a13572baea59b75c6c10100e7c7e212/prospect/utils/smoothing.py#L17)
        '''
        if vdisp <= 0: 
            return wave, flux
        from scipy.ndimage import gaussian_filter1d
        pixkms = 10.0                                 # SSP pixel size [km/s]
        dlogwave = pixkms / 2.998e5 / np.log(10)
        wlog = 10**np.arange(np.log10(wave.min() + 10.), np.log10(wave.max() - 10.), dlogwave)
        flux_wlog = UT.trapz_rebin(wave, flux, xnew=wlog, edges=None)
        # convolve  
        sigma = vdisp / pixkms # in pixels 
        smoothflux = gaussian_filter1d(flux_wlog, sigma=sigma, axis=0)
        return wlog, smoothflux
    
    def avgSFR(self, tt, zred=None, tage=None, dt=1., t0=None, method='trapz'):
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

        t0 : None or float 
            lookback time where you want to evaluate the average SFR. If
            specified, the function returns the average SFR over the range [t0,
            t0-dt]. If None, [tage, tage-dt]
        '''
        if zred is None and tage is None: 
            raise ValueError('specify either zred or tage')
        if zred is not None and tage is not None: 
            raise ValueError('specify either zred or tage')

        _t, _sfh = self.SFH(tt, tage=tage, zred=zred) # get SFH 
        if np.atleast_2d(_t).shape[0] > 1: 
            ts, sfhs = _t, _sfh 
        else: 
            ts = [_t]
            sfhs = [_sfh] 

        avsfrs = [] 
        for tlookback, sfh in zip(ts, sfhs): 

            sfh = np.atleast_2d(sfh) / 1e9 # in units of 10^9 Msun 

            tage = tlookback[-1] 
            assert tage > dt # check that the age of the galaxy is longer than the timescale 

            if t0 is None: 
                # calculate average SFR over the range tcomic [tage, tage - dt]
                # linearly interpolate SFH at tage - dt  
                i0 = np.where(tlookback < dt)[0][-1]  
                i1 = np.where(tlookback > dt)[0][0]

                sfh_t_dt = (sfh[:,i1] - sfh[:,i0]) / (tlookback[i1] - tlookback[i0]) * (dt - tlookback[i0]) + sfh[:,i0]
            
                _t = np.concatenate([tlookback[:i0+1], [dt]]) 
                _sfh = np.concatenate([sfh[:,:i0+1], sfh_t_dt[:,None]], axis=1)
            else: 
                # calculate average SFR over the range tcomic \in [t0, t0-dt]
                if t0 < tage: 
                    # linearly interpolate SFH to t0   
                    i0 = np.where(tlookback < t0)[0][-1]  
                    i1 = np.where(tlookback > t0)[0][0]

                    sfh_t_t0 = (sfh[:,i1] - sfh[:,i0]) / (tlookback[i1] - tlookback[i0]) * (t0 - tlookback[i0]) + sfh[:,i0]
                else: 
                    sfh_t_t0 = np.zeros(sfh.shape[0])
                
                # linearly interpolate SFH to t0 - dt  
                if t0 + dt < tage: 
                    i0 = np.where(tlookback < t0 + dt)[0][-1]  
                    i1 = np.where(tlookback > t0 + dt)[0][0]

                    sfh_t_t0_dt = (sfh[:,i1] - sfh[:,i0]) / (tlookback[i1] - tlookback[i0]) * (t0 + dt - tlookback[i0]) + sfh[:,i0]
                else: 
                    sfh_t_t0_dt = np.zeros(sfh.shape[0])

                i_in = (tlookback < t0 + dt) & (tlookback > t0)
    
                _t = np.concatenate([[t0], tlookback[i_in], [t0 + dt]]) 
                _sfh = np.concatenate([sfh_t_t0[:,None], sfh[:,i_in], sfh_t_t0_dt[:,None]], axis=1)

            # add up the stellar mass formed during the dt time period 
            if method == 'trapz': 
                avsfr = np.trapz(_sfh, _t) / dt
            elif method == 'simps': 
                from scipy.intergrate import simps
                avsfr = simps(_sfh, _t) / dt
            else: 
                raise NotImplementedError
            avsfrs.append(np.clip(avsfr, 0., None))
        
        if len(avsfrs) == 1: 
            return avsfrs[0]
        else: 
            return np.concatenate(avsfrs) 
    
    def Z_MW(self, tt, tage=None, zred=None):
        ''' given theta calculate mass weighted metallicity using the ZH NMF
        bases. 
        '''
        if tage is None and zred is None: 
            raise ValueError("specify either zred or tage") 
        if tage is not None and zred is not None: 
            raise ValueError("specify either zred or tage") 

        theta = self._parse_theta(tt) 
        t, sfh = self.SFH(tt, tage=tage, zred=zred) # get SFH 
        _, zh = self.ZH(tt, tage=tage, zred=zred, tcosmic=t) 

        # mass weighted average
        z_mw = np.trapz(zh * sfh, t) / (10**theta['logmstar']) # np.trapz(sfh, t) should equal tt[:,0] 

        return np.clip(z_mw, 0, np.inf)

    def _load_NMF_bases(self, name='tojeiro.4comp'): 
        ''' read in NMF SFH and ZH bases. These bases are used to reduce the
        dimensionality of the SFH and ZH. 
        '''
        dir_dat = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat') 
        if name == 'tojeiro.4comp': 
            fsfh = os.path.join(dir_dat, 'NMF_2basis_SFH_components_nowgt_lin_Nc4.txt')
            fzh = os.path.join(dir_dat, 'NMF_2basis_Z_components_nowgt_lin_Nc2.txt') 
            ft = os.path.join(dir_dat, 'sfh_t_int.txt') 

            nmf_sfh = np.loadtxt(fsfh) # basis order is jumbled up it should be [2 ,0, 1, 3]
            nmf_zh  = np.loadtxt(fzh) 
            nmf_t   = np.loadtxt(ft) # look back time 

            self._nmf_t_lb_sfh      = nmf_t[::-1] 
            self._nmf_t_lb_zh       = nmf_t[::-1] 
            self._nmf_sfh_basis     = nmf_sfh[:,::-1]
            self._nmf_zh_basis      = nmf_zh[:,::-1]
        elif name in ['tng.4comp', 'tng.6comp']: 
            icomp = int(name.split('.')[-1][0])
            fsfh = os.path.join(dir_dat, 'NMF_basis.sfh.tng%icomp.txt' % icomp) 
            fzh = os.path.join(dir_dat, 'NMF_2basis_Z_components_nowgt_lin_Nc2.txt') 
            ftsfh = os.path.join(dir_dat, 't_sfh.tng%icomp.txt' % icomp) 
            ftzh = os.path.join(dir_dat, 'sfh_t_int.txt') 

            nmf_sfh     = np.loadtxt(fsfh, unpack=True) 
            nmf_zh      = np.loadtxt(fzh) 
            nmf_tsfh    = np.loadtxt(ftsfh) # look back time 
            nmf_tzh     = np.loadtxt(ftzh) # look back time 

            self._nmf_t_lb_sfh      = nmf_tsfh
            self._nmf_t_lb_zh       = nmf_tzh[::-1]
            self._nmf_sfh_basis     = nmf_sfh 
            self._nmf_zh_basis      = nmf_zh[:,::-1]
        else:
            raise NotImplementedError

        self._Ncomp_sfh = self._nmf_sfh_basis.shape[0]
        self._Ncomp_zh = self._nmf_zh_basis.shape[0]
        
        # SFH bases as a function of lookback time 
        self._sfh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    self._nmf_t_lb_sfh, 
                    self._nmf_sfh_basis[i], k=1) 
                for i in range(self._Ncomp_sfh)
                ]
        self._zh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    self._nmf_t_lb_zh, 
                    self._nmf_zh_basis[i], k=1) 
                for i in range(self._Ncomp_zh)]
        return None 

    def _parse_theta(self, tt):
        ''' parse given array of parameter values 
        '''
        tt = np.atleast_2d(tt) 

        assert tt.shape[1] == len(self._parameters) 

        theta = {} 
        for i, param in enumerate(self._parameters): 
            theta[param] = tt[:,i]
        return theta 


class Tau(Model): 
    ''' SPS model where SFH is parameterized using tau models (standard or
    delayed tau) and constant metallicity history.  

    Parameters
    ----------
    burst : bool
        If True include a star bursts in SFH. (default: True) 

    delayed : bool
        If False, use standard tau model. 
        If True, use delayed-tau model. 
        (default: False) 

    emulator : bool
        If True, use emulator rather than running FSPS. Not yet implemented for
        `Tau`. (default: False) 

    cosmo : astropy.comsology object
        specify cosmology


    Notes 
    -----
    * only supports Calzetti+(2000) attenuation curve) and Chabrier IMF. 

    '''
    def __init__(self, burst=True, delayed=False, emulator=False, cosmo=None): 
        self._ssp = None 
        self._burst = burst 
        self._delayed = delayed
        self._emulator = emulator 
        assert not emulator, "emulator not yet implemneted --- coming soon"
        super().__init__(cosmo=cosmo)

    def _fsps(self, tt, tage): 
        ''' FSPS SPS model with tau or delayed-tau model SFH  


        Parameters 
        ----------
        tt : 1-d array
            Parameter FSPS tau model 
            * log M*  
            * e-folding time in Gyr 0.1 < tau < 10^2
            * constant component 0 < C < 1 
            * start time of SFH in Gyr

            if burst = True
            * fraction of mass formed in an instantaneous burst of star formation
            * age of the universe when burst occurs (tburst < tage) 

            * metallicity 
            * Calzetti+(2000) dust index 
        
        tage : float
            age of the galaxy 

        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 


        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A
        '''
        # initialize FSPS StellarPopulation object
        if self._ssp is None: self._ssp_initiate() 
        theta = self._parse_theta(tt) 

        # sfh parameters
        self._ssp.params['tau']      = theta['tau_sfh'] # e-folding time in Gyr 0.1 < tau < 10^2
        self._ssp.params['const']    = theta['const_sfh'] # constant component 0 < C < 1 
        self._ssp.params['sf_start'] = theta['sf_start'] # start time of SFH in Gyr
        if self._burst: 
            self._ssp.params['fburst'] = theta['fburst'] # fraction of mass formed in an instantaneous burst of star formation
            self._ssp.params['tburst'] = theta['tburst'] # age of the universe when burst occurs (tburst < tage) 

        # metallicity
        self._ssp.params['logzsol']  = np.log10(theta['metallicity']/0.0190) # log(z/zsun) 
        # dust 
        self._ssp.params['dust2']    = theta['dust2']  # dust2 parameter in fsps 
        
        w, l_ssp = self._ssp.get_spectrum(tage=tage, peraa=True) 
        
        # mass normalization
        l_ssp *= (10**theta['logmstar']) 

        return w, l_ssp 

    def SFH(self, tt, zred=None, tage=None): 
        ''' tau or delayed-tau star formation history given parameter values.

        Parameters
        ----------
        tt : 1d or 2d array
            Nparam or NxNparam array specifying the parameter values 

        zred : float, optional
            redshift of the galaxy

        tage : float, optional
            age of the galaxy in Gyrs

        Notes
        -----
        * 12/22/2020: decided to have SFH integrate to 1 by using numerically
           integrated normalization rather than analytic  
        * There are some numerical errors where the SFH integrates to slightly
            greater than one. It should be good enough for most purposes. 
        '''
        if zred is None and tage is None: 
            raise ValueError("specify either the redshift or age of the galaxy")
        if tage is None: 
            tage = self.cosmo.age(zred).value # age in Gyr

        from scipy.special import gammainc
        
        theta       = self._parse_theta(tt) 
        logmstar    = theta['logmstar'] 
        tau         = theta['tau_sfh'] 
        const       = theta['const_sfh']
        sf_start    = theta['sf_start']
        if self._burst: 
            fburst  = theta['fburst'] 
            tburst  = theta['tburst'] 
    
        # tau or delayed-tau 
        power = 1 
        if self._delayed: power = 2

        t = np.linspace(sf_start, np.repeat(tage, sf_start.shape[0]), 100).T 
        tlookback = t - sf_start[:,None]
        dt = np.diff(t, axis=1)[:,0]
        
        tmax = (tage - sf_start) / tau
        normalized_t = (t - sf_start[:,None])/tau[:,None]
        
        # constant contribution 
        sfh = (np.tile(const / (tage-sf_start), (100, 1))).T

        # burst contribution 
        if self._burst: 
            tb = (tburst - sf_start) / tau
            has_burst = (tb > 0)
            fburst[~has_burst] = 0. 
            iburst = np.floor(tb[has_burst] / dt[has_burst] * tau[has_burst]).astype(int)
            dts = (np.tile(dt, (100, 1))).T
            dts[:,0] *= 0.5 
            dts[:,-1] *= 0.5 
            sfh[has_burst, iburst] += fburst[has_burst] / dts[has_burst, iburst]
        else: 
            fburst = 0. 

        # tau contribution 
        ftau = (1. - const - fburst) 
        sfh_tau = (normalized_t **(power - 1) * np.exp(-normalized_t))
        sfh += sfh_tau * (ftau / tau / np.trapz(sfh_tau, normalized_t))[:,None]
        #sfh += ftau[:,None] / tau[:,None] * (normalized_t **(power - 1) *
        #        np.exp(-normalized_t) / gammainc(power, tmax.T)[:,None])
        
        # normalize by stellar mass 
        sfh *= 10**logmstar[:,None]

        if np.atleast_2d(tt).shape[0] == 1: 
            return tlookback[0], sfh[0][::-1]
        else: 
            return tlookback, sfh[:,::-1]

    def ZH(self, tt, zred=None, tage=None, tcosmic=None):
        ''' calculate metallicity history. For `Tau` this is simply a constant
        metallicity value  

        Parameters
        ----------
        tt : 1d or 2d array
            Nparam or [N, Nparam] array specifying the parameter values

        zred : float, optional
            redshift of the galaxy

        tage : float
            age of the galaxy in Gyrs 
        '''
        if zred is None and tage is None: 
            raise ValueError("specify either the redshift or age of the galaxy")
        if tage is None: 
            assert isinstance(zred, float)
            tage = self.cosmo.age(zred).value # age in Gyr

        theta = self._parse_theta(tt)
        Z = theta['metallicity'] 

        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 100)
        
        return t, np.tile(np.atleast_2d(Z).T, (1, 100))

    def _init_model(self, **kwargs): 
        ''' some under the hood initalization of the model and its
        parameterization. 
        '''
        if self._burst: 
            self._parameters = [
                    'logmstar', 
                    'tau_sfh',      # e-folding time in Gyr 0.1 < tau < 10^2
                    'const_sfh',    # constant component 0 < C < 1 
                    'sf_start',     # start time of SFH in Gyr #'sf_trunc'    
                    'fburst',       # fraction of mass formed in an instantaneous burst of star formation
                    'tburst',       # age of the universe when burst occurs 
                    'metallicity', 
                    'dust2']
        else: 
            self._parameters = [
                    'logmstar', 
                    'tau_sfh',      # e-folding time in Gyr 0.1 < tau < 10^2
                    'const_sfh',    # constant component 0 < C < 1 
                    'sf_start',     # start time of SFH in Gyr #'sf_trunc',     
                    'metallicity', 
                    'dust2']

        if not self._emulator: 
            self._sps_model = self._fsps 

        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        # tau or delayed tau model
        if not self._delayed: sfh = 1 
        else: sfh = 4 

        dust_type   = 2 # Calzetti et al. (2000) attenuation curve
        imf_type    = 1 # chabrier
        self._ssp = fsps.StellarPopulation(
                zcontinuous=1,          # interpolate metallicities
                sfh=sfh,                # sfh type 
                dust_type=dust_type,            
                imf_type=imf_type)      # chabrier 
        return None  


class NMF(Model): 
    ''' SPS model with SFH and ZH parameterized using NMF bases. 
  
    ** more details to come ** 

    Parameters
    ----------
    burst : bool
        If True include a star bursts in SFH. (default: True) 
    
    emulator : bool
        If True, use emulator rather than running FSPS. 

    cosmo : astropy.comsology object
        specify cosmology


    Notes 
    -----
    * only supports 4 component SFH with or without burst and 2 component ZH 
    * only supports Calzetti+(2000) attenuation curve) and Chabrier IMF. 
    '''
    def __init__(self, burst=True, emulator=False, cosmo=None): 
        self._ssp = None 
        self._burst = burst
        self._emulator = emulator 
        super().__init__(cosmo=cosmo) # initializes the model

    def _emu(self, tt, tage): 
        ''' FSPS emulator model using NMF SFH and ZH bases with the optional
        starburst 

        Parameters 
        ----------
        tt : 1d array 
            Nparam array that specifies the parameter values 

        tage : float 
            age of galaxy 
        
        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 

        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A
    
        to-do
        -----
        * replace redshift dependence of emulator to tage for consistency
        '''
        theta = self._parse_theta(tt) 
        
        assert np.isclose(np.sum([theta['beta1_sfh'], theta['beta2_sfh'],
            theta['beta3_sfh'], theta['beta4_sfh']]), 1.), "SFH basis coefficients should add up to 1"
    
        # get redshift with interpolation 
        zred = self._z_tage_interp(tage) 
    
        tt_nmf = np.concatenate([theta['beta1_sfh'], theta['beta2_sfh'],
            theta['beta3_sfh'], theta['beta4_sfh'], theta['gamma1_zh'],
            theta['gamma2_zh'], theta['dust1'], theta['dust2'],
            theta['dust_index'], [zred]])

        # NMF from emulator 
        lum_ssp = np.exp(self._emu_nmf(tt_nmf)) 
    
        # add burst contribution 
        if self._burst: 
            fburst = theta['fburst']
            tburst = theta['tburst'] 

            lum_burst = np.zeros(lum_ssp.shape)
            # if starburst is within the age of the galaxy 
            if tburst < tage and fburst > 0.: 
                lum_burst = np.exp(self._emu_burst(tt)) 

            # renormalize NMF contribution  
            lum_ssp *= (1. - fburst) 

            # add in burst contribution 
            lum_ssp += fburst * lum_burst

        # normalize by stellar mass 
        lum_ssp *= (10**theta['logmstar'])

        return self._nmf_emu_waves, lum_ssp

    def _fsps(self, tt, tage): 
        ''' FSPS SPS model using NMF SFH and ZH bases 

        Parameters 
        ----------
        tt : 1d array 
            Nparam array that specifies the parameter values 

        tage : float 
            age of galaxy 
        

        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 


        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A

        Notes
        -----
        * 12/23/2020: age of SSPs are no longer set to the center of the
          lookback time as this  ignores the youngest tage ~ 0 SSP, which have
          significant contributions. Also we set a minimum tage = 1e-8 because
          FSPS returns a grid for tage = 0 but 1e-8 gets the youngest isochrone
          anyway. 
        '''
        if self._ssp is None: self._ssp_initiate()  # initialize FSPS StellarPopulation object
        theta = self._parse_theta(tt) 
        
        assert np.isclose(np.sum([theta['beta1_sfh'], theta['beta2_sfh'],
            theta['beta3_sfh'], theta['beta4_sfh']]), 1.), "SFH basis coefficients should add up to 1"
        
        # NMF SFH(t) noramlized to 1 **without burst**
        tlookback, sfh = self.SFH(
                np.concatenate([[0.], tt[1:]]), 
                tage=tage, burst=False)  
        # NMF ZH at lookback time bins 
        _, zh = self.ZH(tt, tage=tage)

        dt = np.zeros(len(tlookback))
        dt[1:-1] = 0.5 * (np.diff(tlookback)[1:] + np.diff(tlookback)[:-1]) 
        dt[0]   = 0.5 * (tlookback[1] - tlookback[0]) 
        dt[-1]  = 0.5 * (tlookback[-1] - tlookback[-2]) 
    
        for i, tage in enumerate(tlookback): 
            m = dt[i] * sfh[i] # mass formed in this bin 
            if m == 0 and i != 0: continue 

            self._ssp.params['logzsol'] = np.log10(zh[i]/0.0190) # log(Z/Zsun)
            self._ssp.params['dust1'] = theta['dust1']
            self._ssp.params['dust2'] = theta['dust2']  
            self._ssp.params['dust_index'] = theta['dust_index']
            
            wave_rest, lum_i = self._ssp.get_spectrum(
                    tage=np.clip(tage, 1e-8, None), 
                    peraa=True) # in units of Lsun/AA
            # note that this spectrum is normalized such that the total formed
            # mass = 1 Msun

            if i == 0: lum_ssp = np.zeros(len(wave_rest))
            lum_ssp += m * lum_i 
    
        # add burst contribution 
        if self._burst: 
            fburst = theta['fburst']
            tburst = theta['tburst'] 

            lum_burst = np.zeros(lum_ssp.shape)
            # if starburst is within the age of the galaxy 
            if tburst < tage: 
                _, lum_burst = self._fsps_burst(tt)

            # renormalize NMF contribution  
            lum_ssp *= (1. - fburst) 

            # add in burst contribution 
            lum_ssp += fburst * lum_burst

        # normalize by stellar mass 
        lum_ssp *= (10**theta['logmstar'])
        return wave_rest, lum_ssp

    def _fsps_burst(self, tt, debug=False):
        ''' dust attenuated luminosity contribution from a SSP that corresponds
        to the burst. This spectrum is normalized such that the total formed
        mass is 1 Msun, **not** fburst 
        '''
        if self._ssp is None: self._ssp_initiate()  # initialize FSPS StellarPopulation object
        theta = self._parse_theta(tt) 
        tt_zh = np.array([theta['gamma1_zh'], theta['gamma2_zh']])

        tburst = theta['tburst'] 

        dust1           = theta['dust1']
        dust2           = theta['dust2']
        dust_index      = theta['dust_index']

        # get metallicity at tburst 
        zburst = np.sum(np.array([tt_zh[i] * self._zh_basis[i](tburst) 
            for i in range(self._N_nmf_zh)]))
        
        if debug:
            print('zburst=%e' % zburst) 
            print('dust1=%f' % dust1) 
            print('dust2=%f' % dust2) 
            print('dust_index=%f' % dust_index) 
    
        # luminosity of SSP at tburst 
        self._ssp.params['logzsol'] = np.log10(zburst/0.0190) # log(Z/Zsun)
        self._ssp.params['dust1'] = dust1
        self._ssp.params['dust2'] = dust2 
        self._ssp.params['dust_index'] = dust_index
        
        wave_rest, lum_burst = self._ssp.get_spectrum(
                tage=np.clip(tburst, 1e-8, None), 
                peraa=True) # in units of Lsun/AA
        # note that this spectrum is normalized such that the total formed
        # mass = 1 Msun
        return wave_rest, lum_burst

    def _emu_nmf(self, tt):
        ''' 
        
        Parameters
        ----------
        tt : 1d array 
            Nparam array that specifies 
            [beta1_sfh, beta2_sfh, beta3_sfh, beta4_sfh, gamma1_zh, gamma2_zh, dust1, dust2, dust_index, redshift] 
    
        Returns
        -------
        logflux : array_like[Nwave,] 
            (natural) log of (SSP luminosity in units of Lsun/A)
        '''
        logflux = [] 

        for iwave in range(self._nmf_n_emu): # wave bins
            W_, b_, alphas_, betas_, parameters_shift_, parameters_scale_,\
                    pca_shift_, pca_scale_, spectrum_shift_, spectrum_scale_,\
                    pca_transform_matrix_, wavelengths, n_layers =\
                    self._nmf_emu_params[iwave] 

            # forward pass through the network
            act = []
            layers = [(tt - parameters_shift_)/parameters_scale_]
            for i in range(n_layers-1):

                # linear network operation
                act.append(np.dot(layers[-1], W_[i]) + b_[i])

                # pass through activation function
                layers.append((betas_[i] + (1.-betas_[i])*1./(1.+np.exp(-alphas_[i]*act[-1])))*act[-1])

            # final (linear) layer -> (normalized) PCA coefficients
            layers.append(np.dot(layers[-1], W_[-1]) + b_[-1])

            # rescale PCA coefficients, multiply out PCA basis -> normalized spectrum, shift and re-scale spectrum -> output spectrum
            logflux.append(np.dot(layers[-1]*pca_scale_ + pca_shift_,
                pca_transform_matrix_)*spectrum_scale_ + spectrum_shift_)

        return np.concatenate(logflux) 
    
    def _emu_burst(self, tt, debug=False): 
        ''' calculate the dust attenuated luminosity contribution from a SSP
        that corresponds to the burst using an emulator. This spectrum is
        normalized such that the total formed mass is 1 Msun, **not** fburst 

        Notes
        -----
        * currently luminosity contribution is set to 0 if tburst > 13.27 due
        to FSPS numerical accuracy  
        '''
        theta = self._parse_theta(tt) 
        tt_zh = np.array([theta['gamma1_zh'], theta['gamma2_zh']])

        tburst = theta['tburst'] 

        if tburst > 13.27: 
            warnings.warn('tburst > 13.27 Gyr returns 0s --- modify priors')
            return np.zeros(len(self._nmf_emu_waves))

        dust1           = theta['dust1']
        dust2           = theta['dust2']
        dust_index      = theta['dust_index']

        # [tburst, ZH coeff0, ZH coeff1, dust1, dust2, dust_index]
        tt = np.array([
            tburst, 
            theta['gamma1_zh'], 
            theta['gamma2_zh'], 
            theta['dust1'], 
            theta['dust2'], 
            theta['dust_index']]).flatten()

        logflux = [] 
        for iwave in range(self._burst_n_emu): # wave bins
            W_, b_, alphas_, betas_, parameters_shift_, parameters_scale_,\
                    pca_shift_, pca_scale_, spectrum_shift_, spectrum_scale_,\
                    pca_transform_matrix_, wavelengths, n_layers =\
                    self._burst_emu_params[iwave] 

            # forward pass through the network
            act = []
            layers = [(tt - parameters_shift_)/parameters_scale_]
            for i in range(n_layers-1):

                # linear network operation
                act.append(np.dot(layers[-1], W_[i]) + b_[i])

                # pass through activation function
                layers.append((betas_[i] + (1.-betas_[i])*1./(1.+np.exp(-alphas_[i]*act[-1])))*act[-1])

            # final (linear) layer -> (normalized) PCA coefficients
            layers.append(np.dot(layers[-1], W_[-1]) + b_[-1])

            # rescale PCA coefficients, multiply out PCA basis -> normalized spectrum, shift and re-scale spectrum -> output spectrum
            logflux.append(np.dot(layers[-1]*pca_scale_ + pca_shift_,
                pca_transform_matrix_)*spectrum_scale_ + spectrum_shift_)
        return np.concatenate(logflux) 

    def _load_emulator(self): 
        ''' read in pickle files that contains the parameters for the FSPS
        emulator that is split into wavelength bins
        '''
        # load NMF emulator 
        npcas = [50, 30, 30]
        f_nn = lambda npca, i: 'fsps.nmf_bases.seed0_499.3w%i.pca%i.8x256.nbatch250.pkl' % (i, npca)
        
        self._nmf_n_emu         = len(npcas)
        self._nmf_emu_params    = [] 
        self._nmf_emu_wave      = [] 

        for i, npca in enumerate(npcas): 
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                f_nn(npca, i)), 'rb')
            params = pickle.load(fpkl)
            
            self._nmf_emu_params.append(params)
            self._nmf_emu_wave.append(params[11])

        self._nmf_emu_waves = np.concatenate(self._nmf_emu_wave) 

        # load burst emulator here once it's done training 
        npcas = [50, 30, 30]
        f_nn = lambda npca, i: 'fsps.burst.seed0_499.3w%i.pca%i.8x256.nbatch250.pkl' % (i, npca)
        self._burst_n_emu = len(npcas)
        self._burst_emu_params    = [] 
        self._burst_emu_wave      = [] 

        for i, npca in enumerate(npcas): 
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                f_nn(npca, i)), 'rb')
            params = pickle.load(fpkl)
            
            self._burst_emu_params.append(params)
            self._burst_emu_wave.append(params[11])
        
        # check that emulator wavelengths agree
        assert np.array_equal(self._nmf_emu_waves, np.concatenate(self._burst_emu_wave))
        return None 

    def SFH(self, tt, zred=None, tage=None, burst=True): 
        ''' star formation history for given set of parameter values and
        redshift. SFH is parameterized using a 4 component SFH NMF bases with
        or without an instantaneous star burst 
    
        Parameters
        ----------
        tt : 1d or 2d array 
            Nparam or NxNparam array of parameter values 

        tage : float
            age of the galaxy 

        burst : bool
            If True include burst. This overrides `burst` kwarg specified in
            __init__

        Returns
        -------
        t : 1d or 2d array 
            linearly spaced look back time

        sfh : 1d or 2d array 
            star formation history at lookback time specified by t 
        '''
        if zred is None and tage is None: 
            raise ValueError("specify either the redshift or age of the galaxy")
        if tage is None: 
            assert isinstance(zred, float)
            tage = self.cosmo.age(zred).value # age in Gyr

        theta = self._parse_theta(tt) 

        # sfh nmf basis coefficients 
        tt_sfh = np.array([theta['beta1_sfh'], theta['beta2_sfh'],
            theta['beta3_sfh'], theta['beta4_sfh']]).T

        t = np.linspace(0., tage, 100) # look back time 
        
        # normalized basis out to t 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) 
            for i in range(self._N_nmf_sfh)])

        # caluclate normalized SFH
        sfh = np.sum(np.array([tt_sfh[:,i][:,None] * _basis[i][None,:] 
            for i in range(self._N_nmf_sfh)]), axis=0)
    
        # add starburst event 
        if self._burst and burst: 
            fburst = theta['fburst'] # fraction of stellar mass from star burst
            tburst = theta['tburst'] # time of star burst
       
            noburst = (tburst > tage)
            fburst[noburst] = 0. 

            # add normalized starburst to SFH 
            sfh *= (1. - fburst)[:,None] 
            sfh += fburst[:,None] * self._SFH_burst(tburst, t)

        # multiply by stellar mass 
        sfh *= 10**theta['logmstar'][:,None]

        if np.atleast_2d(tt).shape[0] == 1: return t, sfh[0]
        return t, sfh 

    def _SFH_burst(self, tburst, tgrid): 
        ''' place a single star-burst event on a given *evenly spaced* lookback time grid
        '''
        tburst = np.atleast_1d(tburst)
        dt = tgrid[1] - tgrid[0] 
        
        sfh = np.zeros((len(tburst), len(tgrid)))

        # burst within the age of the galaxy 
        has_burst = (tburst < tgrid.max()) 

        iburst = np.floor(tburst[has_burst] / dt).astype(int)
        dts = np.repeat(dt, 100)
        dts[0] *= 0.5 
        dts[-1] *= 0.5 
        sfh[has_burst, iburst] += 1. / dts[iburst]
        return sfh 
    
    def ZH(self, tt, zred=None, tage=None, tcosmic=None): 
        ''' metallicity history for given set of parameters. metallicity is
        parameterized using a 2 component NMF basis. The parameter values
        specify the coefficient of the components and this method returns the
        linear combination of the two. 

        parameters
        ----------
        tt : array_like[N,Nparam]
           parameter values of the model 

        zred : float
            redshift of galaxy/csp

        tcosmic : array_like
            cosmic time

        Returns 
        -------
        t : array_like[100,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[N,100]
            metallicity at cosmic time t --- ZH(t) 
        '''
        if zred is None and tage is None: 
            raise ValueError("specify either the redshift or age of the galaxy")
        if tage is None: 
            assert isinstance(zred, float)
            tage = self.cosmo.age(zred).value # age in Gyr
        theta = self._parse_theta(tt) 

        # metallicity history basis coefficients  
        tt_zh = np.array([theta['gamma1_zh'], theta['gamma2_zh']]).T

        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 100)

        # metallicity basis
        _z_basis = np.array([self._zh_basis[i](t) for i in range(self._N_nmf_zh)]) 

        # get metallicity history
        zh = np.sum(np.array([tt_zh[:,i][:,None] * _z_basis[i][None,:] for i in range(self._N_nmf_zh)]), axis=0) 

        if tt_zh.shape[0] == 1: return t, zh[0]
        return t, zh 

    def _init_model(self, **kwargs): 
        ''' some under the hood initalization of the model and its
        parameterization. 
        '''
        # load 4 component NMF bases from Rita 
        self._load_NMF_bases(name='tojeiro.4comp')
        self._N_nmf_sfh = 4 # 4 NMF component SFH
        self._N_nmf_zh  = 2 # 2 NMF component ZH 

        if not self._burst:
            self._parameters = [
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
        else: 
            self._parameters = [
                    'logmstar', 
                    'beta1_sfh', 
                    'beta2_sfh', 
                    'beta3_sfh',
                    'beta4_sfh', 
                    'fburst', 
                    'tburst',   # lookback time of the universe when burst occurs (tburst < tage) 
                    'gamma1_zh', 
                    'gamma2_zh', 
                    'dust1', 
                    'dust2',
                    'dust_index']

        if not self._emulator: 
            self._sps_model = self._fsps 
        else: 
            self._sps_model = self._emu
            self._load_emulator()

        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        sfh         = 0 # tabulated SFH
        dust_type   = 4 # dust1, dust2, and dust_index 
        imf_type    = 1 # chabrier

        self._ssp = fsps.StellarPopulation(
                zcontinuous=1,          # interpolate metallicities
                sfh=sfh,                # sfh type 
                dust_type=dust_type,            
                imf_type=imf_type)             # chabrier 
        return None  


class FSPS(Model): 
    ''' Model class object with FSPS models with tau model SFH parameterizations. 
    '''
    def __init__(self, name='nmf_bases', cosmo=None): 
        self.name = name 
        super().__init__(cosmo=cosmo)
        self._ssp = None 

    def sed(self, tt, zred, vdisp=0., wavelength=None, resolution=None,
            filters=None, debug=False): 
        ''' compute the redshifted spectral energy distribution (SED) for a given set of parameter values and redshift.
       

        Parameters
        ----------
        tt : array_like[Nsample,Nparam] 
            SPS parameters     

        zred : float or array_like
            redshift of the SED 

        vdisp : float or array_like
            velocity dispersion  

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  

        resolution : array_like[N,Nwave]
            resolution matrix (e.g. DESI data provides a resolution matrix)  
    
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
        if self._ssp is None: 
            # initialize FSPS StellarPopulation object
            self._ssp_initiate() 

        tt      = np.atleast_2d(tt)
        zred    = np.atleast_1d(zred) 
        vdisp   = np.atleast_1d(vdisp) 
        ntheta  = tt.shape[1]

        assert tt.shape[0] == zred.shape[0]
       
        outwave, outspec, maggies = [], [], [] 
        for _tt, _zred in zip(tt, zred): 

            if debug: 
                print('FSPS.sed: redshift = %f' % _zred)

            tt_arr = np.concatenate([_tt, [_zred]])
            if debug: print('FSPS.sed: theta', tt_arr)
            
            # get SSP luminosity
            wave_rest, ssp_lum = self._sps_model(tt_arr[1:])
            if debug: print('FSPS.sed: ssp lum', ssp_lum)

            # mass normalization
            lum_ssp = (10**tt_arr[0]) * ssp_lum

            # redshift the spectra
            w_z = wave_rest * (1. + _zred)
            d_lum = self._d_lum_z_interp(_zred) 
            flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + _zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang

            # apply velocity dispersion 
            wave_smooth, flux_smooth = self._apply_vdisp(w_z, flux_z, vdisp)
            
            if wavelength is None: 
                outwave.append(wave_smooth)
                outspec.append(flux_smooth)
            else: 
                outwave.append(wavelength)

                # resample flux to input wavelength  
                resampflux = UT.trapz_rebin(wave_smooth, flux_smooth, xnew=wavelength) 

                if resolution is not None: 
                    # apply resolution matrix 
                    _i = 0 
                    for res in np.atleast_1d(resolution):
                        _res = UT.Resolution(res) 
                        resampflux[_i:_i+res.shape[-1]] = _res.dot(resampflux[_i:_i+res.shape[-1]]) 
                        _i += res.shape[-1]
                outspec.append(resampflux) 

            if filters is not None: 
                # calculate photometry from SEDs 
                _maggies = filters.get_ab_maggies(np.atleast_2d(flux_z) *
                        1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_z *
                        U.Angstrom)
                maggies.append(np.array(list(_maggies[0])) * 1e9)

        if len(outwave) == 1: 
            outwave = outwave[0] 
            outspec = outspec[0] 
            if filters is not None: maggies = maggies[0]
        else: 
            outwave = np.array(outwave)
            outspec = np.array(outspec) 
            if filters is not None: maggies = np.array(maggies)

        if filters is None: 
            return outwave, outspec
        else: 
            return outwave, outspec, maggies

    def _fsps_tau(self, tt): 
        ''' FSPS SPS model with tau or delayed-tau model SFH  


        Parameters 
        ----------
        tt : array_like[Nparam] 
            Parameter FSPS tau model 
        
            * e-folding time in Gyr 0.1 < tau < 10^2
            * constant component 0 < C < 1 
            * start time of SFH in Gyr
            * trunctation time of the SFH in Gyr
            * fraction of mass formed in an instantaneous burst of star formation
            * age of the universe when burst occurs (tburst < tage) 
            * metallicity 
            * Calzetti+(2000) dust index 


        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 


        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A
        '''
        if self._ssp is None: 
            # initialize FSPS StellarPopulation object
            self._ssp_initiate() 

        # sfh parameters
        self._ssp.params['tau']      = tt[0] # e-folding time in Gyr 0.1 < tau < 10^2
        self._ssp.params['const']    = tt[1] # constant component 0 < C < 1 
        self._ssp.params['sf_start'] = tt[2] # start time of SFH in Gyr
        #self._ssp.params['sf_trunc'] = tt[3] # trunctation time of the SFH in Gyr
        self._ssp.params['fburst']   = tt[3] # fraction of mass formed in an instantaneous burst of star formation
        self._ssp.params['tburst']   = tt[4] # age of the universe when burst occurs (tburst < tage) 
        # metallicity
        self._ssp.params['logzsol']  = np.log10(tt[5]/0.0190) # log(z/zsun) 
        # dust 
        self._ssp.params['dust2']    = tt[6] # dust2 parameter in fsps 

        return self._ssp.get_spectrum(tage=tt[7], peraa=True) 

    def SFH(self, tt, zred): 
        ''' star formation history given parameters and redshift 
        '''
        return self._SFH(np.atleast_2d(tt), zred)

    def _SFH_tau(self, tt, zred): 
        ''' tau or delayed-tau model star formation history 

        Notes
        -----
        * 12/22/2020: decided to have SFH integrate to 1 by using numerically
           integrated normalization rather than analytic  
        * There are some numerical errors where the SFH integrates to slightly
            greater than one. It should be good enough for most purposes. 
        '''
        from scipy.special import gammainc
        tage = self.cosmo.age(zred).value 

        tau         = tt[:,1] 
        const       = tt[:,2]
        sf_start    = tt[:,3]
        fburst      = tt[:,4]
        tburst      = tt[:,5]
        # indices in theta 
        if self._delayed_tau: power = 2 
        else: power = 1

        t = np.linspace(sf_start, np.repeat(tage, sf_start.shape[0]), 100).T 
        tlookback = t - sf_start[:,None]
        dt = np.diff(t, axis=1)[:,0]
        
        tb      = (tburst - sf_start) / tau
        tmax    = (tage - sf_start) / tau
        normalized_t = (t - sf_start[:,None])/tau[:,None]
        
        # constant contribution 
        sfh = (np.tile(const / (tage-sf_start), (100, 1))).T

        # burst contribution 
        has_burst = (tb > 0)
        fburst[~has_burst] = 0. 
        iburst = np.floor(tb[has_burst] / dt[has_burst] * tau[has_burst]).astype(int)
        dts = (np.tile(dt, (100, 1))).T
        dts[:,0] *= 0.5 
        dts[:,-1] *= 0.5 
        sfh[has_burst, iburst] += fburst[has_burst] / dts[has_burst, iburst]

        # tau contribution 
        ftau = (1. - const - fburst) 
        sfh_tau = (normalized_t **(power - 1) * np.exp(-normalized_t))
        sfh += sfh_tau * (ftau / tau / np.trapz(sfh_tau, normalized_t))[:,None]
        #sfh += ftau[:,None] / tau[:,None] * (normalized_t **(power - 1) *
        #        np.exp(-normalized_t) / gammainc(power, tmax.T)[:,None])
        
        sfh *= 10**tt[:,0][:,None]
        if tt.shape[0] == 1: 
            return tlookback[0], sfh[0][::-1]
        else: 
            return tlookback, sfh[:,::-1]

    def ZH(self, tt, zred, tcosmic=None):
        ''' metallicity history 
        '''
        return self._ZH(np.atleast_2d(tt), zred, tcosmic=tcosmic) 
    
    def _ZH_tau(self, tt, zred, tcosmic=None): 
        ''' constant metallicity 
        '''
        Z = tt[:,7]

        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 100)
        
        return t, np.tile(np.atleast_2d(Z).T, (1, 100))

    def _init_model(self, **kwargs): 
        ''' initialize theta values 
        '''
        self._sps_parameters = [
                'logmstar', 
                'tau_sfh',      # e-folding time in Gyr 0.1 < tau < 10^2
                'const_sfh',    # constant component 0 < C < 1 
                'sf_start',     # start time of SFH in Gyr #'sf_trunc',     # trunctation time of the SFH in Gyr
                'fburst',       # fraction of mass formed in an instantaneous burst of star formation
                'tburst',       # age of the universe when burst occurs 
                'metallicity', 
                'dust2']
        self._sps_model = self._fsps_tau 
        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        if self.name == 'tau': 
            sfh         = 1
            dust_type   = 2 # Calzetti et al. (2000) attenuation curve
            imf_type    = 1 # chabrier
        elif self.name == 'delayed_tau': 
            sfh         = 4
            dust_type   = 2 # Calzetti et al. (2000) attenuation curve
            imf_type    = 1 # chabrier
        else: 
            raise NotImplementedError

        self._ssp = fsps.StellarPopulation(
                zcontinuous=1,          # interpolate metallicities
                sfh=sfh,                  # sfh type 
                dust_type=dust_type,            
                imf_type=imf_type)             # chabrier 
        return None  


class FSPS_NMF(FSPS): 
    ''' Model class object with FSPS models with NMF basis SFH and ZH parameterizations. 
    Currently supports models `nmf_bases` and `nmfburst`

    **description here** 

    '''
    def __init__(self, name='nmf', cosmo=None): 
        super().__init__(name=name, cosmo=cosmo) # initializes the model

    def _fsps_nmfburst(self, tt): 
        ''' FSPS SPS model using NMF SFH and ZH bases 

        Parameters 
        ----------
        tt : array_like[Nparam] 
            FSPS parameters (does not include stellar mass)

        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 


        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A

        Notes
        -----
        * 12/23/2020: age of SSPs are no longer set to the center of the
          lookback time as this  ignores the youngest tage ~ 0 SSP, which have
          significant contributions. Also we set a minimum tage = 1e-8 because
          FSPS returns a grid for tage = 0 but 1e-8 gets the youngest isochrone
          anyway. 
        * 02/22/2021: restructured so that the NMF and burst contributions are
        separated 
        '''
        fburst, tburst = tt[self._i_sfh_burst-1]
        zred = tt[-1] 
        tage = self.cosmo.age(zred).value # age in Gyr

        # nmf luminosity contribution 
        wave_rest, lum_ssp = self._fsps_nmf(tt)
        
        # starburst luminosity contribution 
        lum_burst = np.zeros(lum_ssp.shape)
        if tburst < tage: # if starburst is within the age of the galaxy 
            _, lum_burst = self._fsps_burst(tt)

            # renormalize NMF contribution  
            lum_ssp *= (1. - fburst) 

            # add in burst contribution 
            lum_ssp += fburst * lum_burst

        return wave_rest, lum_ssp

    def _fsps_nmf(self, tt): 
        ''' FSPS SPS model using NMF SFH and ZH bases 

        Parameters 
        ----------
        tt : array_like[Nparam] 
            FSPS parameters (does not include stellar mass)

        Returns
        -------
        wave_rest : array_like[Nwave] 
            rest-frame wavelength of SSP flux 


        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A

        Notes
        -----
        * 12/23/2020: age of SSPs are no longer set to the center of the
          lookback time as this  ignores the youngest tage ~ 0 SSP, which have
          significant contributions. Also we set a minimum tage = 1e-8 because
          FSPS returns a grid for tage = 0 but 1e-8 gets the youngest isochrone
          anyway. 
        '''
        if self._ssp is None: self._ssp_initiate()  # initialize FSPS StellarPopulation object
        dust1       = tt[-4]
        dust2       = tt[-3]
        dust_index  = tt[-2]
        zred        = tt[-1] 
        
        # NMF SFH at lookback time  
        tlookback, sfh = self._SFH_nmf(np.concatenate([[0.], tt[:-1]]), zred)
        # NMF ZH at lookback time bins 
        _, zh = self.ZH(np.concatenate([[0.], tt[:-1]]), zred)

        dt = np.zeros(len(tlookback))
        dt[1:-1] = 0.5 * (np.diff(tlookback)[1:] + np.diff(tlookback)[:-1]) 
        dt[0]   = 0.5 * (tlookback[1] - tlookback[0]) 
        dt[-1]  = 0.5 * (tlookback[-1] - tlookback[-2]) 
    
        for i, tage in enumerate(tlookback): 
            m = dt[i] * sfh[i] # mass formed in this bin 
            if m == 0 and i != 0: continue 

            self._ssp.params['logzsol'] = np.log10(zh[i]/0.0190) # log(Z/Zsun)
            self._ssp.params['dust1'] = dust1
            self._ssp.params['dust2'] = dust2 
            self._ssp.params['dust_index'] = dust_index
            
            wave_rest, lum_i = self._ssp.get_spectrum(
                    tage=np.clip(tage, 1e-8, None), 
                    peraa=True) # in units of Lsun/AA
            # note that this spectrum is normalized such that the total formed
            # mass = 1 Msun

            if i == 0: lum_ssp = np.zeros(len(wave_rest))
            lum_ssp += m * lum_i 

        return wave_rest, lum_ssp

    def _fsps_burst(self, tt):
        ''' dust attenuated luminosity contribution from a SSP that corresponds
        to the burst. This spectrum is normalized such that the total formed
        mass is 1 Msun, **not** fburst 
        '''
        if self._ssp is None: self._ssp_initiate()  # initialize FSPS StellarPopulation object
        tt_zh           = tt[self._i_zh_nmf - 1]
        fburst, tburst  = tt[self._i_sfh_burst - 1]
        dust1           = tt[-4]
        dust2           = tt[-3]
        dust_index      = tt[-2]

        # get metallicity at tburst 
        zburst = np.sum(np.array([tt_zh[i] * self._zh_basis[i](tburst) for i in range(self._N_nmf_zh)]))
    
        # luminosity of SSP at tburst 
        self._ssp.params['logzsol'] = np.log10(zburst/0.0190) # log(Z/Zsun)
        self._ssp.params['dust1'] = dust1
        self._ssp.params['dust2'] = dust2 
        self._ssp.params['dust_index'] = dust_index
        
        wave_rest, lum_burst = self._ssp.get_spectrum(
                tage=np.clip(tburst, 1e-8, None), 
                peraa=True) # in units of Lsun/AA
        # note that this spectrum is normalized such that the total formed
        # mass = 1 Msun
        return wave_rest, lum_burst

    def SFH(self, tt, zred): 
        ''' star formation history given parameters and redshift 
        '''
        return self._SFH(np.atleast_2d(tt), zred)

    def _SFH_nmfburst(self, tt, zred): 
        ''' SFH based on NMF bases with a burst
        '''
        # fraction of stellar mass from star burst
        fburst, tburst = tt[:,self._i_sfh_burst].T
        
        noburst = (tburst > self.cosmo.age(zred).value) 
        fburst[noburst] = 0. 
    
        # NMF SFH 
        t, sfh_nmf = self._SFH_nmf(tt, zred)
    
        # (normalized starburst SFH) x (stellar mass) 
        sfh_burst = self._SFH_burst(tburst, t) * 10**tt[:,0][:,None]
        
        # NMF + starburst 
        sfh = (1. - fburst)[:,None] * sfh_nmf + fburst[:,None] * sfh_burst 
        
        if tt.shape[0] == 1: return t, sfh[0]
        return t, sfh

    def _SFH_nmf(self, tt, zred): 
        ''' SFH based on NMF SFH bases
        '''
        tt = np.atleast_2d(tt) # necessary redundancy

        # sfh nmf basis coefficients 
        tt_sfh = tt[:,self._i_sfh_nmf] 
        
        assert isinstance(zred, float)

        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0., tage, 100) # look back time 
        
        # normalized basis out to t 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) 
            for i in range(self._N_nmf_sfh)])

        # caluclate normalized SFH
        sfh = np.sum(np.array([tt_sfh[:,i][:,None] * _basis[i][None,:] 
            for i in range(self._N_nmf_sfh)]), axis=0)

        # multiply by stellar mass 
        sfh *= 10**tt[:,0][:,None]

        if tt.shape[0] == 1: return t, sfh[0]
        return t, sfh 
    
    def _SFH_burst(self, tburst, tgrid): 
        ''' place a single star-burst event on a given *evenly spaced* lookback time grid
        '''
        tburst = np.atleast_1d(tburst)
        dt = tgrid[1] - tgrid[0] 
        
        sfh = np.zeros((len(tburst), len(tgrid)))

        # burst within the age of the galaxy 
        has_burst = (tburst < tgrid.max()) 

        iburst = np.floor(tburst[has_burst] / dt).astype(int)
        dts = np.repeat(dt, 100)
        dts[0] *= 0.5 
        dts[-1] *= 0.5 
        sfh[has_burst, iburst] += 1. / dts[iburst]
        return sfh 
    
    def ZH(self, tt, zred, tcosmic=None): 
        ''' metallicity history 

        parameters
        ----------
        tt : array_like[N,Nparam]
           parameter values of the model 

        zred : float
            redshift of galaxy/csp

        tcosmic : array_like
            cosmic time

        Returns 
        -------
        t : array_like[100,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[N,100]
            metallicity at cosmic time t --- ZH(t) 
        '''
        # metallicity history basis coefficients  
        tt_zh = np.atleast_2d(tt)[:,self._i_zh_nmf]

        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 100)

        # metallicity basis is not normalized
        _z_basis = np.array([self._zh_basis[i](t) for i in range(self._N_nmf_zh)]) 

        # get metallicity history
        zh = np.sum(np.array([tt_zh[:,i][:,None] * _z_basis[i][None,:] for i in range(self._N_nmf_zh)]), axis=0) 

        if tt_zh.shape[0] == 1: return t, zh[0]
        return t, zh 

    def _init_model(self, **kwargs): 
        ''' initialize the model
        '''
        # load 4 component NMF bases from Rita 
        self._load_NMF_bases(name='tojeiro.4comp')
        self._N_nmf_sfh = 4 # 4 NMF component SFH
        self._N_nmf_zh  = 2 # 2 NMF component ZH 

        self._sps_model = self._fsps_nmf

        if self.name == 'nmf': 
            self._sps_parameters = [
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
            self._SFH   = self._SFH_nmf
            # indices of SFH and ZH parameters
            self._i_sfh_nmf  = np.array([1, 2, 3, 4])
            self._i_zh_nmf   = np.array([5, 6]) 
        elif self.name == 'nmfburst': 
            self._sps_parameters = [
                    'logmstar', 
                    'beta1_sfh', 
                    'beta2_sfh', 
                    'beta3_sfh',
                    'beta4_sfh', 
                    'fburst', 
                    'tburst',   # lookback time of the universe when burst occurs (tburst < tage) 
                    'gamma1_zh', 
                    'gamma2_zh', 
                    'dust1', 
                    'dust2',
                    'dust_index']
            self._SFH   = self._SFH_nmfburst
            # indices of SFH and ZH parameters
            self._i_sfh_nmf     = np.array([1, 2, 3, 4])
            self._i_sfh_burst   = np.array([5, 6])
            self._i_zh_nmf      = np.array([7, 8]) 
        else: 
            raise NotImplementedError 
        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        sfh         = 0 # tabulated SFH
        dust_type   = 4 # dust1, dust2, and dust_index 
        imf_type    = 1 # chabrier

        self._ssp = fsps.StellarPopulation(
                zcontinuous=1,          # interpolate metallicities
                sfh=sfh,                # sfh type 
                dust_type=dust_type,            
                imf_type=imf_type)             # chabrier 
        return None  


class Emulator_NMF(FSPS_NMF): 
    ''' FSPS emulator that 

    more details to come 


    References
    ----------
    * Alsing et al.(2020) 


    Dev. Notes
    ----------
    '''
    def __init__(self, name='nmfburst', cosmo=None): 
        super().__init__(name=name, cosmo=cosmo)

    def sed(self, tt, zred, vdisp=0., wavelength=None, resolution=None,
            filters=None, debug=False): 
        ''' compute the redshifted spectral energy distribution (SED) for a given set of parameter values and redshift.
       

        Parameters
        ----------
        tt : array_like[Nsample,Nparam] 
            Parameter values of emulator model
            * For DESIemulator.name == 'nmfburst': [log M*, beta1_sfh,
            beta2_sfh, beta3_sfh, beta4_sfh, fburst, tburst, gamma1_zh,
            gamma2_zh, dust1, dust2, dust_index] 

        zred : float or array_like
            redshift of the SED 

        vdisp : float or array_like
            velocity dispersion  

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  

        resolution : array_like[N,Nwave]
            resolution matrix (e.g. DESI data provides a resolution matrix)  
    
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
        vdisp   = np.atleast_1d(vdisp) 
        ntheta  = tt.shape[1]

        assert tt.shape[0] == zred.shape[0]
       
        outwave, outspec, maggies = [], [], [] 
        for _tt, _zred in zip(tt, zred): 
            # check redshift range
            assert (_zred >= 0.) and (_zred < 0.5), "outside of 0.0 <z < 0.5"
            assert np.isclose(np.sum(_tt[1:5]), 1.), "SFH basis coefficients should add up to 1"


            if debug: 
                print('emulator.sed: redshift = %f' % zred)
                print('emulator.sed: tage = %f' % tage) 

            # [parameters] + [zred] 
            tt_arr = np.concatenate([_tt, [_zred]])
            
            # get SSP luminosity
            ssp_log_lum = self._emulator(tt_arr[1:]) 

            # mass normalization
            lum_ssp = np.exp(tt_arr[0] * np.log(10) + ssp_log_lum)

            # redshift the spectra
            w_z = self._emu_waves * (1. + _zred)
            d_lum = self._d_lum_z_interp(_zred) 
            flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + _zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang
            
            # apply velocity dispersion 
            wave_smooth, flux_smooth = self._apply_vdisp(w_z, flux_z, vdisp)
            
            if wavelength is None: 
                outwave.append(wave_smooth)
                outspec.append(flux_smooth)
            else: 
                outwave.append(wavelength)

                # resample flux to input wavelength  
                resampflux = UT.trapz_rebin(wave_smooth, flux_smooth, xnew=wavelength) 

                if resolution is not None: 
                    # apply resolution matrix 
                    _i = 0 
                    for res in np.atleast_1d(resolution):
                        _res = UT.Resolution(res) 
                        resampflux[_i:_i+res.shape[-1]] = _res.dot(resampflux[_i:_i+res.shape[-1]]) 
                        _i += res.shape[-1]
                outspec.append(resampflux) 

            if filters is not None: 
                # calculate photometry from SEDs 
                _maggies = filters.get_ab_maggies(np.atleast_2d(flux_z) *
                        1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_z *
                        U.Angstrom)
                maggies.append(np.array(list(_maggies[0])) * 1e9)

        if len(outwave) == 1: 
            outwave = outwave[0] 
            outspec = outspec[0] 
            if filters is not None: maggies = maggies[0]
        else: 
            outwave = np.array(outwave)
            outspec = np.array(outspec) 
            if filters is not None: maggies = np.array(maggies)

        if filters is None: 
            return outwave, outspec
        else: 
            return outwave, outspec, maggies

    def _emulator(self, tt):
        ''' forward pass through the the three speculator NN wave bins to
        predict SED 
        
        
        Parameters
        ----------
        tt : array_like[Nparam,]
            * For DESIemulator.name == 'nmfburst': 
            [beta1_sfh, beta2_sfh, beta3_sfh, beta4_sfh, fburst, tburst, gamma1_zh, gamma2_zh, dust1, dust2, dust_index, tage] 
    
        Returns
        -------
        logflux : array_like[Nwave,] 
            (natural) log of (SSP luminosity in units of Lsun/A)
        '''
        logflux = [] 
        for iwave in range(self.n_emu): # wave bins

            # forward pass through the network
            act = []
            offset = np.log(np.sum(tt[:4]))

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

    def _load_model_params(self): 
        ''' read in pickle files that contains the parameters for the FSPS
        emulator that is split into wavelength bins
        '''
        if self.name == 'nmfburst': 
            npcas = [30, 60, 30, 30, 30, 30] # wavelength bins 
            f_nn = lambda npca, i: 'fsps.nmfburst.seed0_499.6w%i.pca%i.10x256.nbatch250.pkl' % (i, npca)
        
        self.n_emu = len(npcas) # number of emulator 
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

        for i, npca in enumerate(npcas): 
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                f_nn(npca, i)), 'rb')
            params = pickle.load(fpkl)
            fpkl.close()

            self._emu_W.append(params[0])
            self._emu_b.append(params[1])
            self._emu_alpha.append(params[2])
            self._emu_beta.append(params[3])
            self._emu_theta_mean.append(params[4])
            self._emu_theta_std.append(params[5])
            self._emu_pca_mean.append(params[6])
            self._emu_pca_std.append(params[7])
            self._emu_spec_mean.append(params[8])
            self._emu_spec_std.append(params[9])
            self._emu_pcas.append(params[10])
            self._emu_wave.append(params[13])

            self._emu_n_layers.append(params[16]) # number of network layers

        self._emu_waves = np.concatenate(self._emu_wave) 
        return None 


class DESIemulator(FSPS): 
    ''' FSPS model emulator that is specifically trained for the DESI PROVABGS.

    more details to come 


    References
    ----------
    * Alsing et al.(2020) 


    Dev. Notes
    ----------
    '''
    def __init__(self, name='nmfburst', cosmo=None): 
        self.name = name # name of model 

        super().__init__(name=name, cosmo=cosmo)

    def sed(self, tt, zred, vdisp=0., wavelength=None, resolution=None,
            filters=None, debug=False): 
        ''' compute the redshifted spectral energy distribution (SED) for a given set of parameter values and redshift.
       

        Parameters
        ----------
        tt : array_like[Nsample,Nparam] 
            Parameter values of emulator model
            * For DESIemulator.name == 'nmfburst': [log M*, beta1_sfh,
            beta2_sfh, beta3_sfh, beta4_sfh, fburst, tburst, gamma1_zh,
            gamma2_zh, dust1, dust2, dust_index] 

        zred : float or array_like
            redshift of the SED 

        vdisp : float or array_like
            velocity dispersion  

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  

        resolution : array_like[N,Nwave]
            resolution matrix (e.g. DESI data provides a resolution matrix)  
    
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
        vdisp   = np.atleast_1d(vdisp) 
        ntheta  = tt.shape[1]

        assert tt.shape[0] == zred.shape[0]
       
        outwave, outspec, maggies = [], [], [] 
        for _tt, _zred in zip(tt, zred): 
            # check redshift range
            assert (_zred >= 0.) and (_zred < 0.5), "outside of 0.0 <z < 0.5"
            assert np.isclose(np.sum(_tt[1:5]), 1.), "SFH basis coefficients should add up to 1"


            if debug: 
                print('emulator.sed: redshift = %f' % zred)
                print('emulator.sed: tage = %f' % tage) 

            # [parameters] + [zred] 
            tt_arr = np.concatenate([_tt, [_zred]])
            
            # get SSP luminosity
            ssp_log_lum = self._emulator(tt_arr[1:]) 

            # mass normalization
            lum_ssp = np.exp(tt_arr[0] * np.log(10) + ssp_log_lum)

            # redshift the spectra
            w_z = self._emu_waves * (1. + _zred)
            d_lum = self._d_lum_z_interp(_zred) 
            flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + _zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang
            
            # apply velocity dispersion 
            wave_smooth, flux_smooth = self._apply_vdisp(w_z, flux_z, vdisp)
            
            if wavelength is None: 
                outwave.append(wave_smooth)
                outspec.append(flux_smooth)
            else: 
                outwave.append(wavelength)

                # resample flux to input wavelength  
                resampflux = UT.trapz_rebin(wave_smooth, flux_smooth, xnew=wavelength) 

                if resolution is not None: 
                    # apply resolution matrix 
                    _i = 0 
                    for res in np.atleast_1d(resolution):
                        _res = UT.Resolution(res) 
                        resampflux[_i:_i+res.shape[-1]] = _res.dot(resampflux[_i:_i+res.shape[-1]]) 
                        _i += res.shape[-1]
                outspec.append(resampflux) 

            if filters is not None: 
                # calculate photometry from SEDs 
                _maggies = filters.get_ab_maggies(np.atleast_2d(flux_z) *
                        1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_z *
                        U.Angstrom)
                maggies.append(np.array(list(_maggies[0])) * 1e9)

        if len(outwave) == 1: 
            outwave = outwave[0] 
            outspec = outspec[0] 
            if filters is not None: maggies = maggies[0]
        else: 
            outwave = np.array(outwave)
            outspec = np.array(outspec) 
            if filters is not None: maggies = np.array(maggies)

        if filters is None: 
            return outwave, outspec
        else: 
            return outwave, outspec, maggies

    def _emulator(self, tt):
        ''' forward pass through the the three speculator NN wave bins to
        predict SED 
        
        
        Parameters
        ----------
        tt : array_like[Nparam,]
            * For DESIemulator.name == 'nmfburst': 
            [beta1_sfh, beta2_sfh, beta3_sfh, beta4_sfh, fburst, tburst, gamma1_zh, gamma2_zh, dust1, dust2, dust_index, tage] 
    
        Returns
        -------
        logflux : array_like[Nwave,] 
            (natural) log of (SSP luminosity in units of Lsun/A)
        '''
        logflux = [] 
        for iwave in range(self.n_emu): # wave bins

            # forward pass through the network
            act = []
            offset = np.log(np.sum(tt[:4]))

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

    def _init_model(self, **kwargs): 
        ''' initialize the Speculator model 
        '''
        if self.name == 'nmfburst': 
            self._sps_parameters = [
                    'logmstar', 
                    'beta1_sfh', 
                    'beta2_sfh', 
                    'beta3_sfh',
                    'beta4_sfh', 
                    'fburst', 
                    'tburst',   # lookback time of the universe when burst occurs (tburst < tage) 
                    'gamma1_zh', 
                    'gamma2_zh', 
                    'dust1', 
                    'dust2',
                    'dust_index']
            self._load_NMF_bases(name='tojeiro.4comp')
            self._SFH   = self._SFH_nmfburst
            self._ZH    = self._ZH_nmfburst 
            self._load_model_params()
        else: 
            raise NotImplementedError
        return None 

    def _load_model_params(self): 
        ''' read in pickle files that contains the parameters for the FSPS
        emulator that is split into wavelength bins
        '''
        if self.name == 'nmfburst': 
            npcas = [30, 60, 30, 30, 30, 30] # wavelength bins 
            f_nn = lambda npca, i: 'fsps.nmfburst.seed0_499.6w%i.pca%i.10x256.nbatch250.pkl' % (i, npca)
        
        self.n_emu = len(npcas) # number of emulator 
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

        for i, npca in enumerate(npcas): 
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                f_nn(npca, i)), 'rb')
            params = pickle.load(fpkl)
            fpkl.close()

            self._emu_W.append(params[0])
            self._emu_b.append(params[1])
            self._emu_alpha.append(params[2])
            self._emu_beta.append(params[3])
            self._emu_theta_mean.append(params[4])
            self._emu_theta_std.append(params[5])
            self._emu_pca_mean.append(params[6])
            self._emu_pca_std.append(params[7])
            self._emu_spec_mean.append(params[8])
            self._emu_spec_std.append(params[9])
            self._emu_pcas.append(params[10])
            self._emu_wave.append(params[13])

            self._emu_n_layers.append(params[16]) # number of network layers

        self._emu_waves = np.concatenate(self._emu_wave) 
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
        self._load_NMF_bases(name='tojeiro.4comp')

    def sed(self, tt, zred, vdisp=0., wavelength=None, resolution=None,
            filters=None, debug=False): 
        ''' compute the redshifted spectral energy distribution (SED) for a given set of parameter values and redshift.
       

        Parameters
        ----------
        tt : array_like[Nsample,Nparam] 
            Parameter values that correspond to [log M*, b1SFH, b2SFH, b3SFH,
            b4SFH, g1ZH, g2ZH, 'dust1', 'dust2', 'dust_index']. 
    
        zred : float or array_like
            redshift of the SED 

        vdisp : float or array_like
            velocity dispersion  

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  

        resolution : array_like[N,Nwave]
            resolution matrix (e.g. DESI data provides a resolution matrix)  
    
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
        vdisp   = np.atleast_1d(vdisp) 
        ntheta  = tt.shape[1]

        assert tt.shape[0] == zred.shape[0]
       
        outwave, outspec, maggies = [], [], [] 
        for _tt, _zred in zip(tt, zred): 
            # check redshift range
            assert (_zred >= 0.) and (_zred < 0.5), "outside of 0.0 <z < 0.5"
            assert np.isclose(np.sum(_tt[1:5]), 1.), "SFH basis coefficients should add up to 1"

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

            # mass normalization
            lum_ssp = np.exp(tt_arr[0] * np.log(10) + ssp_log_lum)

            # redshift the spectra
            w_z = self._emu_waves * (1. + _zred)
            d_lum = self._d_lum_z_interp(_zred) 
            flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + _zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang
            
            # apply velocity dispersion 
            wave_smooth, flux_smooth = self._apply_vdisp(w_z, flux_z, vdisp)
            
            if wavelength is None: 
                outwave.append(wave_smooth)
                outspec.append(flux_smooth)
            else: 
                outwave.append(wavelength)

                # resample flux to input wavelength  
                resampflux = UT.trapz_rebin(wave_smooth, flux_smooth, xnew=wavelength) 

                if resolution is not None: 
                    # apply resolution matrix 
                    _i = 0 
                    for res in np.atleast_1d(resolution):
                        _res = UT.Resolution(res) 
                        resampflux[_i:_i+res.shape[-1]] = _res.dot(resampflux[_i:_i+res.shape[-1]]) 
                        _i += res.shape[-1]
                outspec.append(resampflux) 

            if filters is not None: 
                # calculate photometry from SEDs 
                _maggies = filters.get_ab_maggies(np.atleast_2d(flux_z) *
                        1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_z *
                        U.Angstrom)
                maggies.append(np.array(list(_maggies[0])) * 1e9)

        if len(outwave) == 1: 
            outwave = outwave[0] 
            outspec = outspec[0] 
            if filters is not None: maggies = maggies[0]
        else: 
            outwave = np.array(outwave)
            outspec = np.array(outspec) 
            if filters is not None: maggies = np.array(maggies)

        if filters is None: 
            return outwave, outspec
        else: 
            return outwave, outspec, maggies

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
        tt : array_like[N,Nparam]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float
            redshift

        Returns 
        -------
        t : array_like[100,]
            age of the galaxy, linearly spaced from 0 to the age of the galaxy

        sfh : array_like[N,100]
            star formation history at cosmic time t --- SFH(t) --- in units of
            Msun/**Gyr**. np.trapz(sfh, t) == Mstar 
        '''
        tt = np.atleast_2d(tt)
        tt_sfh = tt[:,1:5] # sfh basis coefficients 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0, tage, 100)
        
        # normalized basis out to t 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) 
            for i in range(4)])

        # caluclate normalized SFH
        sfh = np.sum(np.array([tt_sfh[:,i][:,None] * _basis[i][None,:] for i in range(4)]), axis=0)

        # multiply by stellar mass 
        sfh *= 10**tt[:,0][:,None]

        if tt.shape[0] == 1: 
            return t, sfh[0]
        else: 
            return t, sfh 
    
    def ZH(self, tt, zred, tcosmic=None):
        ''' metallicity history for a set of parameter values `tt` and redshift `zred`

        parameters
        ----------
        tt : array_like[N,Nparam]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float
            redshift

        Returns 
        -------
        t : array_like[100,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[N,100]
            metallicity at cosmic time t --- ZH(t) 
        '''
        tt = np.atleast_2d(tt)
        tt_zh = tt[:,5:7] # zh bases 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 100)
    
        # metallicity basis is not normalized
        _z_basis = np.array([self._zh_basis[i](t) for i in range(2)]) 

        # get metallicity history
        zh = np.sum(np.array([tt_zh[:,i][:,None] * _z_basis[i][None,:] for i in range(2)]), axis=0) 
        if tt.shape[0] == 1: 
            return t, zh[0]
        else: 
            return t, zh 

    def _init_model(self, **kwargs): 
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
