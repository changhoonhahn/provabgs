'''

inference framework 


'''

# --- provabgs --- 
from .models import DESIspeculator


class desiMCMC(object): 
    '''
    '''
    def __init__(self, model=None): 

        if model is None: # default Model class object 
            self.model = DESIspeculator()

        # initiate p(SSFR) 
        self.ssfr_prior = None 
    
    def run(self, wave_obs=None, flux_obs=None, flux_ivar_obs=None,
            photo_obs=None, photo_ivar_obs=None, zred=None, prior=None,
            mask=None, bands=None, specphoto_calib=None, nwalkers=100,
            burnin=100, niter=1000, maxiter=200000, opt_maxiter=100,
            writeout=None, overwrite=False, silent=True): 
        ''' run MCMC using `emcee` to infer the posterior distribution of the
        model parameters given spectroscopy and/or photometry. The function 
        outputs a dictionary with the median theta of the posterior as well as 
        the 1sigma and 2sigma errors on the parameters (see below).

        Parameters
        ----------
        wave_obs : array_like[Nwave,]
            observed wavelength
        
        flux_obs : array_like[Nwave,]
            observed flux __in units of ergs/s/cm^2/Ang__

        flux_ivar_obs : array_like[Nwave,]
            observed flux **inverse variance**. Not uncertainty!
        
        photo_obs : array_like[Nband,]
             observed photometric flux __in units of nanomaggies__

        photo_ivar_obs : array_like[Nband,]
            observed photometric flux **inverse variance**. Not uncertainty!

        zred : float 
            redshift of the observations  
    
        prior : Prior object
            A prior object

        mask : string or array_like[Nwave,]
            boolean array specifying where to mask the spectra. There are a few
            preset masks: 
            * mask == 'emline' masks around emission lines at 3728., 4861.,
            5007., 6564. Angstroms. 
            (Default: None) 

        bands : string
            photometric bands of the photometry data. If 'desi', sets up
            default DESI photometric bands.  
            (Default: None)  

        specphoto_calib: string
            specifies the prescription for combining spectroscopy and
            photometry.  

        nwalkers : int 
            number of walkers. 
            (Default: 100) 
        
        burnin : int 
            number of iterations for burnin. 
            (Default: 100) 
        
        niter : int
            number of iterations. 
            (Default: 1000) purposely set low. 
        
        maxiter : int 
            maximum number of iterations for adaptive method. MCMC can
            always be restarted so, keep this at some sensible number.  
            (default: 100000) 

        opt_maxiter : int
            maximum number of iterations for initial optimizer. 
            (Default: 1000) 
        
        writeout : string 
            string specifying the output file. If specified, everything in the
            output dictionary is written out as well as the entire MCMC chain.
            (Default: None) 

        silent : boolean
            If False, a bunch of messages will be shown. Otherwise it'll remain
            silent 
    
        
        Returns 
        -------
        output : dict
            ouptut is structured into a dictionary the includes the following keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_spec_model'] : flux of best-fit model spectrum
            - output['flux_photo_model'] : flux of best-fit model photometry 
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_spec_data'] : flux of observed spectrum 
            - output['flux_spec_ivar_data'] = inverse variance of the observed flux. 
            - output['flux_photo_data'] : flux of observed photometry 
            - output['flux_photo_viar_data'] :  inverse variance of observed photometry 
        '''
        # check inputs
        obs_data_type = self._obs_data_type(wave_obs, flux_obs, flux_ivar_obs,
                photo_obs, photo_ivar_obs) 

        if 'spec' in obs_data_type: 
            # check mask for spectra 
            _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 
        
        if 'photo' in obs_data_type: 
            # get photometric bands  
            bands_list = self._get_bands(bands)
            assert len(bands_list) == len(photo_obs) 
            # get filters
            filters = specFilter.load_filters(*tuple(bands_list))

        # posterior function args and kwargs
        lnpost_args = (
                wave_obs, 
                flux_obs,               # 10^-17 ergs/s/cm^2/Ang
                flux_ivar_obs,          # 1/(10^-17 ergs/s/cm^2/Ang)^2
                photo_obs,              # nanomaggies
                photo_ivar_obs,         # 1/nanomaggies^2
                zred) 
        lnpost_kwargs = {
                'mask': _mask,          # emission line mask 
                'filters': filters,
                'obs_data_type': obs_data_type, 
                'prior': prior          # prior
                }
        
        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost, 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers,
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter, 
                silent=silent,
                writeout=writeout, 
                overwrite=overwrite) 
        return output  

    def _lnPost(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, mask=None, filters=None, bands=None,
            obs_data_type=None, prior=None, debug=False): 
        ''' log Posterior of parameters `tt_arr` given observed a spectrum
        and/or photometry. 
    

        Parameters
        ----------
        tt_arr : array_like[Nparam,]
            array of free parameters. last element is fspectrophoto factor 

        :param wave_obs:
            wavelength of 'observations'
        :param flux_obs: 
            flux of the observed spectra
        :param flux_ivar_obs :
            inverse variance of of the observed spectra
        :param photo_obs: 
            flux of the observed photometry maggies  
        :param photo_ivar_obs :
            inverse variance of of the observed photometry  
        :param zred:
            redshift of the 'observations'
        :param mask: (optional) 
            A boolean array that specifies what part of the spectra to mask 
            out. (default: None) 
        :param prior: (optional) 
            callable prior object 
        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior
        '''
        if obs_data_type is None: 
            obs_data_type = self._obs_data_type(wave_obs, flux_obs, flux_ivar_obs,
                    photo_obs, photo_ivar_obs) 

        lp = prior._lnPrior(tt_arr) # log prior
        if debug: print('desiMCMC._lnPost: log Prior = %f' % lp) 

        if not np.isfinite(lp): 
            return -np.inf

        chi_tot = self._Chi2_spectrophoto(tt_arr[:-1], wave_obs, flux_obs,
                flux_ivar_obs, photo_obs, photo_ivar_obs, zred, mask=mask,
                f_fiber=tt_arr[-1], filters=filters, bands=bands, 
                dirichlet_transform=dirichlet_transform, debug=debug) 

        return lp - 0.5 * chi_tot

    def _Chi2(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, mask=None, f_fiber=1., filters=None,
            bands=None, dirichlet_transform=False, debug=False): 
        ''' calculated the chi-squared between the data and model spectra. 
        '''
        # model(theta) 
        flux, photo = self._model_spectrophoto(tt_arr, zred=zred,
                wavelength=wave_obs, filters=filters, bands=bands, 
                dirichlet_transform=dirichlet_transform, debug=debug) 
        # data - model(theta) with masking 
        dflux = (f_fiber * flux[~mask] - flux_obs[~mask]) 
        # calculate chi-squared for spectra
        _chi2_spec = np.sum(dflux**2 * flux_ivar_obs[~mask]) 
        if debug: 
            print('iSpeculator._Chi2_spectrophoto: Spectroscopic Chi2 = %f' % _chi2_spec)
        # data - model(theta) for photometry  
        dphoto = (photo - photo_obs) 
        # calculate chi-squared for photometry 
        _chi2_photo = np.sum(dphoto**2 * photo_ivar_obs) 
        if debug: 
            print('iSpeculator._Chi2_spectrophoto: Photometric Chi2 = %f' % _chi2_photo)

        if debug: print('iSpeculator._Chi2_spectrophoto: total Chi2 = %f' %
                (_chi2_spec + _chi2_photo))
        return _chi2_spec + _chi2_photo 

    def _lnPost(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred,
            mask=None, prior=None, dirichlet_transform=False, debug=False): 
        ''' calculate the log posterior for a spectrum

        :param tt_arr: 
            array of free parameters

        :param wave_obs:
            wavelength of 'observations'

        :param flux_obs: 
            flux of the observed spectra

        :param flux_ivar_obs :
            inverse variance of of the observed spectra

        :param zred:
            redshift of the 'observations'

        :param mask: (optional) 
            A boolean array that specifies what part of the spectra to mask 
            out. (default: None) 

        :param prior: (optional) 
            callable prior object. (default: None) 

        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior (default: False) 
    
        :param debug: (optional)
            If True, print debug statements (default: False) 
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if not np.isfinite(lp): 
            return -np.inf
        
        if debug: print('iSpeculator._lnPost: log Prior = %f' % lp) 

        chi_tot = self._Chi2(tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred,
                mask=mask, dirichlet_transform=dirichlet_transform, debug=debug)

        return lp - 0.5 * chi_tot

    def _obs_data_type(self, wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs):
        ''' check the data type of the observations: spectroscopy and/or
        photometry
        '''
        if flux_obs is None and photo_obs is None: 
            raise ValueError("please provide either observed spectra or photometry") 
        elif flux_obs is not None and photo_obs is None: 
            data_type = 'spec'
        elif flux_obs is None and photo_obs is not None: 
            data_type = 'photo'
        elif flux_obs is not None and photo_obs is not None: 
            data_type = 'specphoto'
        
        if flux_obs is not None and flux_ivar_obs is None: 
            raise ValueError("please provide inverse variances for the spectra") 

        if photo_obs is not None and photo_ivar_obs is None: 
            raise ValueError("please provide inverse variances for the photometry") 

        return data_type 

    def _check_mask(self, mask, wave_obs, flux_ivar_obs, zred): 
        ''' check that mask is sensible and mask out any parts of the 
        spectra where the ivar doesn't make sense. 
        '''
        if mask is None: 
            _mask = np.zeros(len(wave_obs)).astype(bool) 

        elif mask == 'emline': 
            # mask 20As around emission lines.
            w_lines = np.array([3728., 4861., 5007., 6564.]) * (1. + zred) 
            
            _mask = np.zeros(len(wave_obs)).astype(bool) 
            for wl in w_lines: 
                inemline = ((wave_obs > wl - 20) & (wave_obs < wl + 20))
                _mask = (_mask | inemline)

        elif isinstance(mask, np.ndarray): 
            # user input array 
            assert np.array_equal(mask, mask.astype(bool))
            assert len(mask) == len(wave_obs) 
            _mask = mask 
        else: 
            raise ValueError("mask = None, 'emline', or boolean array") 

        zero_err = ~np.isfinite(flux_ivar_obs)
        _mask = _mask | zero_err 
        return _mask 


class calib_SpecPhoto(object): 
    ''' 
    '''
