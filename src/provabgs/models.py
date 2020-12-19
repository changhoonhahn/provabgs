'''


module for sps models 


'''
import os 
import h5py 
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
        pass 

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
    
    def avgSFR(self, tt, zred, dt=1., t0=None, method='trapz'):
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
        _t, _sfh = self.SFH(tt, zred) # get SFH 
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
                assert tage > t0 
                # linearly interpolate SFH to t0   
                i0 = np.where(tlookback < t0)[0][-1]  
                i1 = np.where(tlookback > t0)[0][0]

                sfh_t_t0 = (sfh[:,i1] - sfh[:,i0]) / (tlookback[i1] - tlookback[i0]) * (t0 - tlookback[i0]) + sfh[:,i0]
                
                # linearly interpolate SFH to t0 - dt  
                i0 = np.where(tlookback < t0 + dt)[0][-1]  
                i1 = np.where(tlookback > t0 + dt)[0][0]

                sfh_t_t0_dt = (sfh[:,i1] - sfh[:,i0]) / (tlookback[i1] - tlookback[i0]) * (t0 + dt - tlookback[i0]) + sfh[:,i0]

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
    
    def Z_MW(self, tt, zred):
        ''' given theta calculate mass weighted metallicity using the ZH NMF
        bases. 
        '''
        tt = np.atleast_2d(tt) 
        t, sfh = self.SFH(tt, zred) # get SFH 
        _, zh = self.ZH(tt, zred, tcosmic=t) 
    
        # mass weighted average
        z_mw = np.trapz(zh * sfh, t) / (10**tt[:,0]) # np.trapz(sfh, t) should equal tt[:,0] 

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

        elif name == 'tng.6comp': 
            fsfh = os.path.join(dir_dat, 'NMF_basis.sfh.tng6comp.txt') 
            fzh = os.path.join(dir_dat, 'NMF_2basis_Z_components_nowgt_lin_Nc2.txt') 
            ftsfh = os.path.join(dir_dat, 't_sfh.tng6comp.txt') 
            ftzh = os.path.join(dir_dat, 'sfh_t_int.txt') 

            nmf_sfh     = np.loadtxt(fsfh, unpack=True) 
            nmf_zh      = np.loadtxt(fzh) 
            nmf_tsfh    = np.loadtxt(ft) # look back time 
            nmf_tzh     = np.loadtxt(ft) # look back time 

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


class FSPS(Model): 
    ''' Model class object with FSPS models with different SFH
    parameterizations. 
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

            # get tage
            tage = self._tage_z_interp(_zred)

            if debug: 
                print('FSPS.sed: redshift = %f' % zred)
                print('FSPS.sed: tage = %f' % tage) 

            tt_arr = np.concatenate([_tt, [tage]])
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

    def _fsps_nmf(self, tt): 
        ''' FSPS SPS model using NMF SFH and ZH bases 

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
        if self._ssp is None: 
            # initialize FSPS StellarPopulation object
            self._ssp_initiate() 
    
        ncomp_sfh = self._Ncomp_sfh
        ncomp_zh = self._Ncomp_zh
        tt_sfh          = tt[:ncomp_sfh] 
        tt_zh           = tt[ncomp_sfh:ncomp_sfh+ncomp_zh]
        tt_dust1        = tt[ncomp_sfh+ncomp_zh]
        tt_dust2        = tt[ncomp_sfh+ncomp_zh+1]
        tt_dust_index   = tt[ncomp_sfh+ncomp_zh+2]
        tage            = tt[ncomp_sfh+ncomp_zh+3] 
    
        tlookback = np.linspace(0, tage, 50) # lookback time edges
        tages = 0.5 * (tlookback[1:] + tlookback[:-1])
        dt = np.diff(tlookback)

        # SFH at lookback time  
        sfh = np.sum(np.array([
            tt_sfh[i] *
            self._sfh_basis[i](tlookback)/np.trapz(self._sfh_basis[i](tlookback), tlookback) 
            for i in range(ncomp_sfh)]), 
            axis=0)
        # ZH at the center of the lookback time bins 
        zh = np.sum(np.array([
            tt_zh[i] * self._zh_basis[i](tages) for i in range(ncomp_zh)]), 
            axis=0)
    
        for i, tage in enumerate(tages): 
            m = dt[i] * 0.5 * (sfh[i] + sfh[i+1]) # mass formed in this bin 
            if m <= 0 and i != 0: continue 

            self._ssp.params['logzsol'] = np.log10(zh[i]/0.0190) # log(Z/Zsun)
            self._ssp.params['dust1'] = tt_dust1
            self._ssp.params['dust2'] = tt_dust2 
            self._ssp.params['dust_index'] = tt_dust_index
            
            wave_rest, lum_i = self._ssp.get_spectrum(tage=tage, peraa=True) # in units of Lsun/AA
            # note that this spectrum is normalized such that the total formed
            # mass = 1 Msun

            if i == 0: lum_ssp = np.zeros(len(wave_rest))
            lum_ssp += m * lum_i 

        return wave_rest, lum_ssp

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
        tt = np.atleast_2d(tt)
        if self.name in ['nmf_bases', 'nmf_tng6']: return self._SFH_nmf(tt, zred)
        elif self.name == 'nmf_comp': return self._SFH_nmf_compressed(tt, zred)
        elif self.name in ['tau', 'delayed_tau']: return self._SFH_tau(tt, zred)

    def _SFH_nmf(self, tt, zred): 
        ''' SFH based on NMF SFH bases
        '''
        ncomp_sfh = self._Ncomp_sfh
        tt_sfh = tt[:,1:ncomp_sfh+1] # sfh basis coefficients 
        
        assert isinstance(zred, float)

        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0., tage, 50) # look back time 
        
        # normalized basis out to t 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) 
            for i in range(ncomp_sfh)])

        # caluclate normalized SFH
        sfh = np.sum(np.array([tt_sfh[:,i][:,None] * _basis[i][None,:] 
            for i in range(ncomp_sfh)]), axis=0)

        # multiply by stellar mass 
        sfh *= 10**tt[:,0][:,None]

        if tt.shape[0] == 1: 
            return t, sfh[0]
        else: 
            return t, sfh 

    def _SFH_nmf_compressed(self, tt, zred): 
        ''' SFH based on *compressed* NMF SFH bases. Unlike `self._SFH_nmf`,
        the bases do not truncate at tage but the entire basis is
        compressed to fit the range 0-tage.  
        '''
        ncomp_sfh = self._Ncomp_sfh
        tt_sfh = tt[:,1:ncomp_sfh+1] # sfh basis coefficients 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0, tage, 50)
        
        # normalized basis out to t 
        #_basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) 
        #    for i in range(4)])
        _basis = np.array([self._nmf_sfh_basis[i] / np.trapz(self._nmf_sfh_basis[i], t) 
            for i in range(ncomp_sfh)])

        # caluclate normalized SFH
        sfh = np.sum(np.array([tt_sfh[:,i][:,None] * _basis[i][None,:] 
            for i in range(ncomp_sfh)]), axis=0)

        # multiply by stellar mass 
        sfh *= 10**tt[:,0][:,None]

        if tt.shape[0] == 1: 
            return t, sfh[0]
        else: 
            return t, sfh 

    def _SFH_tau(self, tt, zred): 
        ''' tau or delayed-tau model star formation history 

        Notes
        -----
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

        t = np.linspace(np.zeros(sf_start.shape[0]), sf_start, 50).T

        tb      = (tburst - sf_start) / tau
        tmax    = (tage - sf_start) / tau
        normalized_t = (t - sf_start[:,None])/tau[:,None]
        
        # constant contribution 
        sfh = (np.tile(const / 50, (50, 1))).T
        # tau contribution 
        ftau = (1. - const - fburst) 
        sfh += ftau[:,None] * (normalized_t **(power - 1) *
                np.exp(-normalized_t) / gammainc(power, tmax.T)[:,None])
        # burst contribution 
        has_burst = np.where(tb > 0)[0] 
        iburst = np.floor(tb[has_burst] / (tmax[has_burst]/50.)).astype(int)
        sfh[has_burst, iburst] += fburst[has_burst]

        sfh *= 10**tt[:,0][:,None]
        if tt.shape[0] == 1: 
            return t[0], sfh[0]
        else: 
            return t, sfh 

    def ZH(self, tt, zred, tcosmic=None):
        ''' metallicity history 
        '''
        tt = np.atleast_2d(tt)
        if self.name == 'nmf_bases': return self._ZH_nmf(tt, zred, tcosmic=tcosmic)
        elif self.name == 'nmf_comp': return self._ZH_nmf_compressed(tt, zred, tcosmic=tcosmic)
        elif self.name in ['tau', 'delayed_tau']: 
            return self._ZH_tau(tt, zred, tcosmic=tcosmic)

    def _ZH_nmf(self, tt, zred, tcosmic=None): 
        ''' metallicity history for a set of parameter values `tt` and redshift `zred`

        parameters
        ----------
        tt : array_like[N,Nparam]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float
            redshift

        tcosmic : array_like
            cosmic time

        Returns 
        -------
        t : array_like[50,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[N,50]
            metallicity at cosmic time t --- ZH(t) 
        '''
        ncomp_zh = self._Ncomp_zh
        tt_zh = tt[:,self._Ncomp_sfh+1:self._Ncomp_sfh+ncomp_zh+1] # zh bases 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 50)

        # metallicity basis is not normalized
        _z_basis = np.array([self._zh_basis[i](t) for i in range(ncomp_zh)]) 

        # get metallicity history
        zh = np.sum(np.array([tt_zh[:,i][:,None] * _z_basis[i][None,:] 
            for i in range(ncomp_zh)]), axis=0) 
        if tt.shape[0] == 1: 
            return t, zh[0]
        else: 
            return t, zh 

    def _ZH_nmf_compressed(self, tt, zred, tcosmic=None): 
        ''' metallicity history for a set of parameter values `tt` and redshift `zred`

        parameters
        ----------
        tt : array_like[N,Nparam]
           Parameter values of [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH,
           'dust1', 'dust2', 'dust_index']. 

        zred : float
            redshift

        tcosmic : array_like
            cosmic time

        Returns 
        -------
        t : array_like[50,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[N,50]
            metallicity at cosmic time t --- ZH(t) 
        '''
        ncomp_zh = self._Ncomp_zh
        tt_zh = tt[:,self._Ncomp_sfh+1:self._Ncomp_sfh+ncomp_zh+1] # zh bases 
        
        assert isinstance(zred, float)
        tage = self.cosmo.age(zred).value # age in Gyr
        if tcosmic is not None: 
            assert tage >= np.max(tcosmic) 
            t = tcosmic.copy() 
        else: 
            t = np.linspace(0, tage, 50)

        # get metallicity history
        zh = np.sum(np.array([tt_zh[:,i][:,None] *
            self._nmf_zh_basis[i][None,:] for i in range(ncomp_zh)]), axis=0) 
        if tt.shape[0] == 1: 
            return t, zh[0]
        else: 
            return t, zh 
    
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
            t = np.linspace(0, tage, 50)
        
        return t, np.tile(np.atleast_2d(Z).T, (1, 50))

    def _init_model(self, **kwargs): 
        ''' initialize theta values 
        '''
        if self.name in ['nmf_bases', 'nmf_comp']: 
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
            self._sps_model = self._fsps_nmf
            self._load_NMF_bases(name='tojeiro.4comp')
        elif self.name in ['nmf_tng6', 'nmf_tng6_comp']: # 6 components for SFH 
            self._sps_parameters = [
                    'logmstar', 
                    'beta1_sfh', 
                    'beta2_sfh', 
                    'beta3_sfh',
                    'beta4_sfh', 
                    'beta5_sfh', 
                    'beta6_sfh', 
                    'gamma1_zh', 
                    'gamma2_zh', 
                    'dust1', 
                    'dust2',
                    'dust_index']
            self._sps_model = self._fsps_nmf
            self._load_NMF_bases('tng.6comp')
        elif self.name in ['tau', 'delayed_tau']: 
            self._sps_parameters = [
                    'logmstar', 
                    'tau_sfh',      # e-folding time in Gyr 0.1 < tau < 10^2
                    'const_sfh',    # constant component 0 < C < 1 
                    'sf_start',     # start time of SFH in Gyr #'sf_trunc',     # trunctation time of the SFH in Gyr
                    'fburst',       # fraction of mass formed in an instantaneous burst of star formation
                    'tburst',       # age of the universe when burst occurs (tburst < tage) 
                    'metallicity', 
                    'dust2']
            self._sps_model = self._fsps_tau 
            if self.name == 'tau': self._delayed_tau = False
            elif self.name == 'delayed_tau': self._delayed_tau = True
        else: 
            raise NotImplementedError 
        
        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        import fsps
        if self.name in ['nmf_bases', 'nmf_comp']: 
            sfh         = 0 
            dust_type   = 4
            imf_type    = 1 # chabrier
        elif self.name == 'tau': 
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
        t : array_like[50,]
            age of the galaxy, linearly spaced from 0 to the age of the galaxy

        sfh : array_like[N,50]
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
        t : array_like[50,]
            cosmic time linearly spaced from 0 to the age of the galaxy

        zh : array_like[N,50]
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
            t = np.linspace(0, tage, 50)
    
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
