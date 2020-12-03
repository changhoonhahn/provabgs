'''

inference framework 


'''
import os 
import numpy as np 
import zeus 
import emcee
import scipy.stats as stat
import scipy.optimize as op
from speclite import filters as specFilter


class MCMC(object): 
    ''' base class object for MCMC inference 

    Parameters
    ----------
    prior : Prior class object
        `provabgs.infer.Prior` class object 
    '''
    def __init__(self): 
        pass

    def lnPost(self, theta, *args, debug=False, **kwargs):
        ''' log Posterior of parameters `theta` 


        Parameters
        ----------
        theta : array_like[Nparam,]
            parameter values
        '''
        lp = self.prior.lnPrior(theta) # log prior
        if debug: print('  log Prior = %f' % lp) 
        if not np.isfinite(lp): 
            return -np.inf

        # transformed theta. Some priors require transforming the parameter
        # space for sampling (e.g. Dirichlet). For most priors, this
        # transformation will return the same value  
        ttheta = self.prior.transform(theta) 
    
        # calculate likelihood 
        lnlike = self.lnLike(ttheta, *args, debug=debug, **kwargs)

        return lp  + lnlike 
    
    def _emcee(self, lnpost_args, lnpost_kwargs, prior, nwalkers=100,
            burnin=100, niter='adaptive', maxiter=200000, opt_maxiter=1000, 
            debug=False, writeout=None, overwrite=False, **kwargs): 
        ''' Run MCMC, using `emcee` for a given log posterior function.
    
        
        Parameters
        ----------
        lnpost_args : tuple 
            arguments for log posterior function, `self.lnPost`

        lnpost_kwargs : dictionary
            keyward arguments for log posterior function, `self.lnPost`
        
        prior : PriorSeq object
            PriorSeq object which specifies the prior. See `infer.load_priors`. 

        nwalkers : int
            number of mcmc walkers
            (Default: 100) 

        burnin : int 
            number of iterations for burnin. If using `niter='adaptive'`, this
            choice is not particularly important. 
            (Default: 100) 

        niter : int
            number of MCMC iterations. If `niter=adaptive`, MCMC will use an
            adpative method based on periodic evaluations of the Gelman-Rubin
            diagnostic to assess convergences (recommended). 
            (Default: 'adaptive') 

        maxiter : int 
            maximum number of total MCMC iterations for adaptive method. MCMC can
            always be restarted so, keep this at some sensible number.  
            (Default: 100000) 

        opt_maxiter : 
            maximum number of iterations for the initial optimizer. This
            optimization is used to set the initial positions of the walkers. 
            (Default: 1000) 

        debug : boolean 
            If True, debug messages will be printed out 
            (Default: False)  

        writeout : None or str
            name of the writeout files that will be passed into temporary saving function

        overwrite : boolean 
            Set as True if you're overwriting an exisiting MCMC. Otherwise, it
            will append to the MCMC file. 

        Notes:
        -----
        '''
        self.nwalkers = nwalkers # number of walkers 

        ndim = prior.ndim

        if (writeout is None) or (not os.path.isfile(writeout)) or (overwrite): 
            # if mcmc chain file does not exist or we want to overwrite
            # initialize the walkers 
            start = self._initialize_walkers(lnpost_args, lnpost_kwargs, prior,
                    nwalkers=nwalkers, opt_maxiter=opt_maxiter, debug=debug)

            # burn in 
            if debug: print('--- burn-in ---') 
            pos, prob, state = self.sampler.run_mcmc(start, burnin)
            self.sampler.reset()

        else: 
            # file exists and we are appending to it. check that priors and
            # parameters agree. 
            if debug: print('--- continuing from %s ---' % writeout) 

            mcmc = self.read_chain(writeout, flat=False, debug=debug) 

            assert np.array_equal(mcmc['prior_range'].T[0], prior.min), 'prior range does not agree with existing chain'
            assert np.array_equal(mcmc['prior_range'].T[1], prior.max), 'prior range does not agree with existing chain'

            # check that theta names agree 
            #for theta in self.theta_names: 
            #    assert theta in mcmc['theta_names'][...].astype(str), 'parameters are different than existing chain' 

            # get walker position from MCMC file 
            pos = mcmc['mcmc_chain'][-1,:,:] 
            
            # initial sampler 
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnPost, 
                    args=lnpost_args, kwargs=lnpost_kwargs)

        if debug: print('--- running main MCMC ---') 
        if niter == 'adaptive': # adaptive MCMC 
            # ACM interval
            STEP = 1000
            index = 0
            _niter = 0 
            autocorr = np.empty(maxiter)
            
            old_tau = np.inf

            for sample in self.sampler.sample(pos, iterations=maxiter,
                    progress=False, skip_initial_state_check=True):

                if self.sampler.iteration % STEP: continue

                if debug: print('  chain %i' % (index+1))
                tau = self.sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
            
                # check convergence 
                convergent = np.all(tau * 11 < self.sampler.iteration)
                convergent &= np.all(np.abs(old_tau - tau) / tau < 0.05)

                if convergent: 
                    if debug: print(' chains have converged!') 
                    break

                old_tau = tau
                
                _chain = self.sampler.get_chain()

                # write out incrementally
                output = self._save_chains(
                    _chain[_niter:,:,:], 
                    lnpost_args, 
                    lnpost_kwargs,
                    writeout=writeout,
                    overwrite=[False, overwrite][overwrite & (index == 1)],
                    debug=debug, 
                    **kwargs) 
                _niter = _chain.shape[0]
        else:
            # run standard mcmc with niter iterations 
            assert isinstance(niter, int) 
            self.sampler.run_mcmc(pos, niter)
            _chain = self.sampler.get_chain()
                    
            output = self._save_chains(
                _chain,
                lnpost_args, 
                lnpost_kwargs, 
                writeout=writeout,
                overwrite=overwrite, 
                debug=debug, 
                **kwargs) 

        return output 

    def _zeus(self, lnpost_args, lnpost_kwargs, prior, nwalkers=100, niter=1000,
            burnin=100, opt_maxiter=1000, debug=False,
            writeout=None, overwrite=False, **kwargs): 
        ''' sample posterior distribution using `zeus`
    
        
        Parameters
        ----------
        lnpost_args : tuple 
            arguments for log posterior function, `self.lnPost`

        lnpost_kwargs : dictionary
            keyward arguments for log posterior function, `self.lnPost`
        
        prior : PriorSeq object
            PriorSeq object which specifies the prior. See `infer.load_priors`. 

        nwalkers : int
            number of mcmc walkers
            (Default: 100) 

        niter : int
            number of zeus steps
            (Default: 1000) 

        debug : boolean 
            If True, debug messages will be printed out 
            (Default: False)  

        writeout : None or str
            name of the writeout files that will be passed into temporary saving function

        overwrite : boolean 
            Set as True if you're overwriting an exisiting MCMC. Otherwise, it
            will append to the MCMC file. 

        Notes:
        -----
        '''
        self.nwalkers = nwalkers # number of walkers 

        if prior is not None: 
            self.prior = prior
        
        start = self._initialize_walkers(lnpost_args, lnpost_kwargs, prior,
                nwalkers=nwalkers, opt_maxiter=opt_maxiter, debug=debug)

        if debug: print('--- burn-in ---') 
        zeus_sampler = zeus.EnsembleSampler(
                self.nwalkers,
                prior.ndim, 
                self.lnPost, 
                args=lnpost_args, 
                kwargs=lnpost_kwargs)
        zeus_sampler.run_mcmc(start, burnin)
        burnin = zeus_sampler.get_chain()

        if debug: print('--- running main MCMC ---') 
        zeus_sampler = zeus.EnsembleSampler(
                self.nwalkers,
                prior.ndim, 
                self.lnPost, 
                args=lnpost_args, 
                kwargs=lnpost_kwargs)
        zeus_sampler.run_mcmc(burnin[-1], niter)
        _chain = zeus_sampler.get_chain()
                    
        output = self._save_chains(
            _chain,
            lnpost_args, 
            lnpost_kwargs, 
            writeout=writeout,
            overwrite=overwrite, 
            debug=debug, 
            **kwargs) 
        return output 

    def _initialize_walkers(self, lnpost_args, lnpost_kwargs, prior,
            nwalkers=100, opt_maxiter=1000, debug=False):
        ''' initalize the walkers by minimizing the -2 * lnPost
        '''
        # initialize the walkers 
        if debug: print('--- initializing the walkers ---') 
        ndim = prior.ndim 

        # initialize optimiser 
        x0 = np.mean([prior.sample() for i in range(10)], axis=0)
        std0 = np.std([prior.sample() for i in range(10)], axis=0)
    
        _lnpost = lambda *args: -2. * self.lnPost(*args, **lnpost_kwargs) 
        min_result = op.minimize(
                _lnpost, 
                x0, 
                args=lnpost_args, 
                method='Nelder-Mead', 
                options={'maxiter': opt_maxiter}) 
        tt0 = min_result['x'] 
        if debug: print('initial theta = [%s]' % ', '.join([str(_t) for _t in tt0])) 
        # initial walker positions 
        p0 = [tt0 + 0.1 * std0 * np.random.randn(ndim) for i in range(nwalkers)]
        # chekc that they're within the prior
        for i in range(nwalkers): 
            while not np.isfinite(self.prior.lnPrior(p0[i])): 
                p0[i] = tt0 + 1e-3 * std0 * np.random.randn(ndim)
        return p0

    def _save_chains(self, chain, lnpost_args, lnpost_kwargs, writeout=None,
            overwrite=False, debug=False, **kwargs):
        ''' save MC chains to file along with other details. If file exists, it
        will append it to the hdf5 file unless `overwrite=True`. 
        '''
        if writeout is None: 
            pass
        elif not overwrite and os.path.isfile(writeout): 
            if debug: print('--- appending to %s ---' % writeout)

            # read existing chain 
            _mcmc = self.read_chain(writeout)  
            old_chain = _mcmc['mcmc_chain']

            assert old_chain.shape[1] == chain.shape[1]
            assert old_chain.shape[2] == chain.shape[2]
            
            # append chain to existing mcmc
            mcmc = h5py.File(writeout, 'a')  #  append 
            mcmc.create_dataset('mcmc_chain%i' % _mcmc['nchain'], data=chain)

            chain = np.concatenate([old_chain, chain], axis=0) 
            newfile = False
        else:   
            if not silent: print('  writing to ... %s' % writeout)
            mcmc = h5py.File(writeout, 'w')  # write 
            mcmc.create_dataset('mcmc_chain0', data=chain) # first chain 
            newfile = True
    
        # get summary of the posterior from *all* of the chains in file 
        flat_chain = self._flatten_chain(chain)
        lowlow, low, med, high, highhigh = np.percentile(flat_chain, 
                [2.5, 16, 50, 84, 97.5], axis=0)
    
        output = {} 
        output['theta_med'] = med 
        output['theta_1sig_plus'] = high
        output['theta_2sig_plus'] = highhigh
        output['theta_1sig_minus'] = low
        output['theta_2sig_minus'] = lowlow
        
        if writeout is None:
            output['mcmc_chain'] = chain 
            return output

        if not newfile: 
            # update these columns
            for k in output.keys(): 
                mcmc[k][...] = output[k]
        else: 
            # writeout these columns
            for k in output.keys(): 
                mcmc.create_dataset(k, data=output[k]) 
        mcmc.close() 
        output['mcmc_chain'] = chain 
        return output  

    def read_chain(self, fchain, flat=False, debug=False): 
        ''' read MCMC file. MCMC will be saved in chunks. This method reads in
        all the chunks and appends them together into one big chain 

        Parameters
        ----------
        fchain : str
            name of the MCMC file 

        flat : boolean 
            If True, flatten the chain using `self._flatten_chain`
            (Default: False) 

        debug : boolean 
            If True, print debug messages
            (Default: False) 
        '''
        if debug: print('--- reading %s --- ' % fchain) 

        chains = h5py.File(fchain, 'r') 
    
        mcmc = {} 
        i_chains = []
        for k in chains.keys(): 
            if 'mcmc_chain' in k: 
                i_chains.append(int(k.replace('mcmc_chain', '')))
            else: 
                mcmc[k] = chains[k][...]
         
        nchain = np.max(i_chains)+1 # number of chains 
        mcmc['nchain'] = nchain

        chain_dsets = []
        for i in range(nchain):
            chain_dsets.append(chains['mcmc_chain%i' % i]) 
        if debug: print('  %i chains read' % nchain) 

        if not flat: 
            mcmc['mcmc_chain'] = np.concatenate(chain_dsets, axis=0) 
        else:
            mcmc['mcmc_chain'] = self._flatten_chain(np.concatenate(chain_dsets, axis=0)) 
        chains.close() 
        return mcmc 

    def _flatten_chain(self, chain): 
        ''' flatten mcmc chain. If chain object is 2D then it assumes it's
        already flattened. 
        '''
        if len(chain.shape) == 2: return chain # already flat 

        s = list(chain.shape[1:])
        s[0] = np.prod(chain.shape[:2]) 
        return chain.reshape(s)


class desiMCMC(MCMC): 
    ''' MCMC inference object specifically designed for analyzing DESI
    spectroscopic and photometric data.
    '''
    def __init__(self, model=None, flux_calib=None, prior=None): 
        if model is None: # default Model class object 
            from .models import DESIspeculator
            self.model = DESIspeculator()
        else: 
            self.model = model

        if flux_calib is None: # default FluxCalib function  
            from .flux_calib import no_flux_factor 
            self.flux_calib = no_flux_factor
        else: 
            self.flux_calib = flux_calib

        self.prior = prior 
    
    def run(self, wave_obs=None, flux_obs=None, flux_ivar_obs=None,
            photo_obs=None, photo_ivar_obs=None, zred=None, prior=None,
            mask=None, bands=None, sampler='emcee', nwalkers=100, niter=1000,
            burnin=100, maxiter=200000, opt_maxiter=100, writeout=None,
            overwrite=False, debug=False, **kwargs): 
        ''' run MCMC using `emcee` to infer the posterior distribution of the
        model parameters given spectroscopy and/or photometry. The function 
        outputs a dictionary with the median theta of the posterior as well as 
        the 1sigma and 2sigma errors on the parameters (see below).

        Parameters
        ----------
        wave_obs : array_like[Nwave,] or list 
            observed wavelengths. Or list of wavelengths
        
        flux_obs : array_like[Nwave,] or list 
            observed flux __in units of ergs/s/cm^2/Ang__. 

        flux_ivar_obs : array_like[Nwave,] or [Narm, Nwave] 
            observed flux **inverse variance**. Not uncertainty!
        
        photo_obs : array_like[Nband,]
             observed photometric flux __in units of nanomaggies__

        photo_ivar_obs : array_like[Nband,]
            observed photometric flux **inverse variance**. Not uncertainty!

        zred : float 
            redshift of the observations  
    
        prior : PriorSeq object
            A `infer.PriorSeq` object

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

        nwalkers : int 
            number of walkers. 
            (Default: 100) 
        
        niter : int
            number of iterations. 
            (Default: 1000) purposely set low. 
        
        burnin : int 
            number of iterations for burnin. 
            (Default: 100) 
        
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

        debug : boolean
            If True, print debug messages
    
        
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
        lnpost_args, lnpost_kwargs = self._lnPost_args_kwargs(
                wave_obs=wave_obs, flux_obs=flux_obs, flux_ivar_obs=flux_ivar_obs,
                photo_obs=photo_obs, photo_ivar_obs=photo_ivar_obs, zred=zred,
                prior=prior, mask=mask, bands=bands)
        
        # run MCMC 
        if sampler == 'emcee': 
            mcmc_sampler = self._emcee
        elif sampler == 'zeus': 
            mcmc_sampler = self._zeus

        output = mcmc_sampler( 
                lnpost_args, 
                lnpost_kwargs, 
                self.prior,
                nwalkers=nwalkers,
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter, 
                writeout=writeout, 
                overwrite=overwrite, 
                debug=debug) 
        return output  

    def lnLike(self, tt, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, mask=None, filters=None, obs_data_type=None,
            debug=False):
        ''' calculated the log likelihood. 
        '''
        # separate SED parameters from Flux Calibration parameters
        tt_sed, tt_fcalib = self.prior.separate_theta(tt, 
                labels=['sed', 'flux_calib'])
        
        # calculate SED model(theta) 
        if isinstance(wave_obs, list): 
            # list of wavelengths are provided e.g. b, r, z spectrograph
            _sed = self.model.sed(tt_sed, zred, wavelength=np.concatenate(wave_obs), filters=filters, debug=debug)
            if 'photo' in obs_data_type: _, _flux, photo = _sed
            else: _, _flux = _sed
            
            # get model fluxes for each wavelength 
            Nwaves = np.cumsum([0]+[len(_w) for _w in wave_obs]) 
            _flux = [_flux[Nwaves[i]:Nwaves[i+1]] for i in range(len(wave_obs))] 

        else: 
            _sed = self.model.sed(tt_sed, zred,
                    wavelength=wave_obs, filters=filters, debug=debug)
            if 'photo' in obs_data_type: _, _flux, photo = _sed
            else: _, _flux = _sed

        _chi2_spec, _chi2_photo = 0., 0.
        if 'spec' in obs_data_type: 
            # apply flux calibration model
            flux = self.flux_calib(tt_fcalib, _flux)

            # data - model(theta) with masking 
            if isinstance(wave_obs, list): 
                dflux = (np.concatenate(flux)[~mask] - np.concatenate(flux_obs)[~mask]) 
                flux_ivar_obs = np.concatenate(flux_ivar_obs) 
            else: 
                dflux = (flux[~mask] - flux_obs[~mask]) 

            # calculate chi-squared for spectra
            _chi2_spec = np.sum(dflux**2 * flux_ivar_obs[~mask]) 

            if debug: print('desiMCMC.lnLike: Spectroscopic Chi2 = %f' % _chi2_spec)

        if 'photo' in obs_data_type: 
            # data - model(theta) for photometry  
            dphoto = (photo - photo_obs) 
            # calculate chi-squared for photometry 
            _chi2_photo = np.sum(dphoto**2 * photo_ivar_obs) 
            if debug: print('desiMCMC.lnLike: Photometric Chi2 = %f' % _chi2_photo)

        if debug: print('desiMCMC.lnLike: total Chi2 = %f' % (_chi2_spec + _chi2_photo))

        chi2 = _chi2_spec + _chi2_photo
        return -0.5 * chi2

    def _lnPost_args_kwargs(self, wave_obs=None, flux_obs=None, flux_ivar_obs=None,
            photo_obs=None, photo_ivar_obs=None, zred=None, prior=None,
            mask=None, bands=None):
        ''' get arg and kwargs for lnPost method 
        '''
        # check inputs
        obs_data_type = self._obs_data_type(wave_obs, flux_obs, flux_ivar_obs,
                photo_obs, photo_ivar_obs) 

        if prior is not None: # update prior if specified  
            self.prior = prior 
    
        assert 'sed' in self.prior.labels, 'please label which priors are for the SED'

        # check mask for spectra 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 
        
        # get photometric bands  
        filters = self._get_bands(bands)

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
                'obs_data_type': obs_data_type
                }
        return lnpost_args, lnpost_kwargs
    
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
        if wave_obs is None: 
            return None 
        elif mask is None: 
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

    def _get_bands(self, bands): 
        ''' For specific `bands`, get the corresponding photometric bandpass
        filters. 

        
        Returns
        -------
        filters : object 
            `speclite.FilterResponse` that correspond to the specified `bands.
        '''
        if bands is None: 
            return None

        if isinstance(bands, str): 
            if bands == 'desi': 
                bands_list = ['decam2014-g', 'decam2014-r', 'decam2014-z']
                #, 'wise2010-W1', 'wise2010-W2']#, 'wise2010-W3', 'wise2010-W4']
            else: 
                raise NotImplementedError("specified bands not implemented") 
        elif isinstance(bands, list): 
            bands_list = bands
        else: 
            raise NotImplementedError("specified bands not implemented") 

        return specFilter.load_filters(*tuple(bands_list))
   
    def _save_chains(self, chain, lnpost_args, lnpost_kwargs, writeout=None,
            overwrite=False, debug=False, **kwargs):
        output = super()._save_chains(chain, lnpost_args, lnpost_kwargs,
                writeout=writeout, overwrite=overwrite, debug=debug)

        obs_data_type = lnpost_kwargs['obs_data_type']
    
        wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred = lnpost_args

        output['redshift']              = zred
        output['wavelength_obs']        = wave_obs
        output['flux_spec_obs']         = flux_obs
        output['flux_ivar_spec_obs']    = flux_ivar_obs
        output['flux_photo_obs']        = photo_obs
        output['flux_ivar_photo_obs']   = photo_ivar_obs

        # add best-fit model to output dictionary
        ttheta = self.prior.transform(output['theta_med'])
        tt_sed, tt_fcalib = self.prior.separate_theta(ttheta, labels=['sed', 'flux_calib'])
        
        # calculate SED model(theta) 
        _sed = self.model.sed(tt_sed, zred, wavelength=wave_obs,
                filters=lnpost_kwargs['filters'], debug=debug)
        if 'photo' in obs_data_type: 
            _, _flux, photo = _sed
        else: _, _flux = _sed

        if 'photo' in obs_data_type: 
            output['flux_photo_model']  = photo
        if 'spec' in obs_data_type: 
            # apply flux calibration model
            flux = self.flux_calib(tt_fcalib, _flux).flatten()
            output['flux_spec_model']   = flux 

        return output 


# --- priors --- 
def load_priors(list_of_prior_obj): 
    ''' load list of `infer.Prior` class objects into a PriorSeq object

    ***MORE DOCUMENTATION HERE***
    '''
    return PriorSeq(list_of_prior_obj)


class PriorSeq(object):
    ''' immutable sequence of priors that are assumed to be statistically
    independent. 

    Parmaeter
    ---------
    list_of_priors : array_like[Npriors,]
        list of `infer.Prior` objects

    '''
    def __init__(self, list_of_priors): 
        self.list_of_priors = list_of_priors 

    def lnPrior(self, theta): 
        ''' evaluate the log prior at theta 
        '''
        theta = np.atleast_2d(theta)

        i = 0 
        lnp_theta = 0. 
        for prior in self.list_of_priors: 
            _lnp_theta = prior.lnPrior(theta[:,i:i+prior.ndim]) 
            if not np.isfinite(_lnp_theta): return -np.inf
            
            lnp_theta += _lnp_theta 
            i += prior.ndim

        return lnp_theta 

    def sample(self): 
        ''' sample the prior 
        '''
        samp = [] 
        for prior in self.list_of_priors: 
            samp.append(prior.sample())
        return np.concatenate(samp) 
    
    def transform(self, tt): 
        ''' transform theta 
        '''
        tt = np.atleast_2d(tt)
        tt_p = np.empty(tt.shape) 

        i = 0 
        for prior in self.list_of_priors: 
            tt_p[:,i:i+prior.ndim] = prior.transform(tt[:,i:i+prior.ndim])
            i += prior.ndim
        return tt_p 

    def separate_theta(self, theta, labels=None): 
        ''' separate theta based on label
        '''
        theta = np.atleast_2d(theta)

        lbls = np.concatenate([np.repeat(prior.label, prior.ndim) 
            for prior in self.list_of_priors]) 

        output = [] 
        for lbl in labels: 
            islbl = (lbls == lbl) 
            output.append(theta[:,islbl])
        return output 

    def append(self, another_list_of_priors): 
        ''' append more Prior objects to the sequence 
        '''
        # join list 
        self.list_of_priors += another_list_of_priors 
        return None 

    @property
    def ndim(self): 
        # update ndim  
        return np.sum([prior.ndim for prior in self.list_of_priors]) 
    
    @property
    def labels(self): 
        return np.array([prior.label for prior in self.list_of_priors]) 


class Prior(object): 
    ''' base class for prior
    '''
    def __init__(self, label=None):
        self.ndim = None 
        self.label = label 
        self._random = np.random.mtrand.RandomState()

    def transform(self, tt): 
        ''' Some priors require transforming the parameter space for sampling
        (e.g. Dirichlet priors). For most priors, this transformation will return 
        the same value  
        '''
        return tt 

    def lnPrior(self, theta):
        ''' evaluate the log prior at theta
        '''
        return 0.


class FlatDirichletPrior(Prior): 
    ''' flat dirichlet prior
    '''
    def __init__(self, ndim, label=None):
        super().__init__(label=label)
        self.ndim = ndim 

    def transform(self, tt): 
        ''' warped manifold transformation as specified in Betancourt (2013).
        This function transforms samples from a uniform distribution to a
        Dirichlet distribution .

        x_i = (\prod\limits_{k=1}^{i-1} z_k) * f 

        f = 1 - z_i         for i < m
        f = 1               for i = m 
    
        Parameters
        ----------
        tt : array_like[N,m]
            N samples drawn from a m-dimensional uniform distribution 

        Returns
        -------
        tt_d : array_like[N,m]
            N transformed samples drawn from a m-dimensional dirichlet
            distribution 

        Reference
        ---------
        * Betancourt(2013) - https://arxiv.org/pdf/1010.3436.pdf
        '''
        tt      = np.atleast_2d(tt)
        assert self.ndim == tt.shape[1]
        tt_d    = np.empty(tt.shape) 
    
        tt_d[:,0] = 1. - tt[:,0]
        for i in range(1,self.ndim-1): 
            tt_d[:,i] = np.prod(tt[:,:i], axis=1) * (1. - tt[:,i]) 
        tt_d[:,-1] = np.prod(tt[:,:-1], axis=1) 
        return tt_d 

    def lnPrior(self, theta):
        ''' evaluate the prior at theta. We assume theta here is
        *untransformed* --- i.e. sampled from a uniform distribution. 

        Parameters
        ----------
        theta : array_like[m,]
            m-dimensional set of parameters 

        Return
        ------
        prior : float
            prior evaluated at theta
        '''
        if np.all(theta <= np.ones(self.ndim)) and np.all(theta >= np.zeros(self.ndim)): 
            return 0.
        else: 
            return -np.inf
    
    def append(self, *arg, **kwargs): 
        raise ValueError("appending not supproted") 

    def sample(self): 
        return np.array([self._random.uniform(0., 1.) for i in range(self.ndim)])


class UniformPrior(Prior): 
    ''' uniform tophat prior
    
    '''
    def __init__(self, _min, _max, label=None):
        super().__init__(label=label)
        
        self.min = np.atleast_1d(_min)
        self.max = np.atleast_1d(_max)
        self.ndim = self.min.shape[0]
        assert self.min.shape[0] == self.max.shape[0]
        assert np.all(self.min < self.max)
        
    def lnPrior(self, theta):
        if np.all(theta < self.max) and np.all(theta >= self.min): 
            return 0.
        else:
            return -np.inf

    def sample(self): 
        return np.array([self._random.uniform(mi, ma) for (mi, ma) in zip(self.min, self.max)])


class GaussianPrior(Prior): 
    ''' Gaussian prior 
    '''
    def __init__(self, mean, covariance, label=None):
        super().__init__(label=label)
        
        self.mean = np.atleast_1d(mean)
        self.covariance = np.atleast_1d(covariance)
        self.ndim = self.mean.shape[0]
        assert self.mean.shape[0] == self.covariance.shape[0]
        self.multinorm = stat.multivariate_normal(self.mean, self.covariance)
    
    def lnPrior(self, theta):
        return self.multinorm.logpdf(theta) 

    def sample(self): 
        return np.atleast_1d(self.multinorm.rvs())
