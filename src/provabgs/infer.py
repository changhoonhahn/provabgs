'''

inference framework 


'''
import os 
import h5py 
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

    def lnPrior(self, theta, debug=False): 
        ''' log Prior with `corrprior.CorrectPrior` prior correction support
        '''
        lp = self.prior.lnPrior(theta)
        if not np.isfinite(lp): 
            return -np.inf

        if self.corrprior is not None: 
            # transform theta 
            ttheta = self.prior.transform(theta)
            # get derived properties
            fttheta = self.corrprior._get_properties(ttheta)
            
            # p'(theta)  = q(theta)/p(f(theta)|q)
            lpf = self.corrprior.p_ftheta.log_pdf(fttheta)
            if not np.isfinite(lpf): 
                return -np.inf
            lp -= lpf 

        if debug: print('  log Prior = %f' % lp) 
        return lp 

    def lnPost(self, theta, *args, debug=False, **kwargs):
        ''' log Posterior of parameters `theta` 


        Parameters
        ----------
        theta : array_like[Nparam,]
            parameter values
        '''
        lp = self.lnPrior(theta, debug=debug) # log prior
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
    
    def _emcee(self, lnpost_args, lnpost_kwargs, nwalkers=100,
            burnin=100, niter='adaptive', maxiter=200000, opt_maxiter=1000, 
            debug=False, writeout=None, overwrite=False, **kwargs): 
        ''' Run MCMC, using `emcee` for a given log posterior function.
    
        
        Parameters
        ----------
        lnpost_args : tuple 
            arguments for log posterior function, `self.lnPost`

        lnpost_kwargs : dictionary
            keyward arguments for log posterior function, `self.lnPost`
        
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

        ndim = self.prior.ndim

        if (writeout is None) or (not os.path.isfile(writeout)) or (overwrite): 
            # if mcmc chain file does not exist or we want to overwrite
            # initialize the walkers 
            start = self._initialize_walkers(lnpost_args, lnpost_kwargs, 
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

            assert np.array_equal(mcmc['prior_range'].T[0], self.prior.min), 'prior range does not agree with existing chain'
            assert np.array_equal(mcmc['prior_range'].T[1], self.prior.max), 'prior range does not agree with existing chain'

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
                # apply prior transform (e.g. this would transform SFH bases into
                # Dirichlet priors) 
                chain = self.prior.transform(_chain[_niter:,:,:])

                # write out incrementally
                output = self._save_chains(
                        chain, 
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

            # apply prior transform (e.g. this would transform SFH bases into
            # Dirichlet priors) 
            chain = self.prior.transform(_chain)

            output = self._save_chains(
                    chain,
                    lnpost_args, 
                    lnpost_kwargs, 
                    writeout=writeout,
                    overwrite=overwrite, 
                    debug=debug, 
                    **kwargs) 

        return output 

    def _zeus(self, lnpost_args, lnpost_kwargs, nwalkers=100, niter=1000,
            burnin=100, opt_maxiter=1000, debug=False, writeout=None,
            overwrite=False, theta_start=None, progress=True, pool=None,
            **kwargs): 
        ''' sample posterior distribution using `zeus`
    
        
        Parameters
        ----------
        lnpost_args : tuple 
            arguments for log posterior function, `self.lnPost`

        lnpost_kwargs : dictionary
            keyward arguments for log posterior function, `self.lnPost`
        
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

        self.ndim_sampling = self.prior.ndim_sampling
        
        start = self._initialize_walkers(lnpost_args, lnpost_kwargs, 
                nwalkers=nwalkers, opt_maxiter=opt_maxiter,
                theta_start=theta_start, debug=debug)

        if debug: print('--- running MCMC ---') 
        zeus_sampler = zeus.EnsembleSampler(
                self.nwalkers,
                self.ndim_sampling, 
                self.lnPost, 
                args=lnpost_args, 
                kwargs=lnpost_kwargs,  
                pool=pool)
        zeus_sampler.run_mcmc(start, burnin + niter, progress=progress)

        _chain = zeus_sampler.get_chain()[burnin:,:,:]

        # apply prior transform (e.g. this would transform SFH bases into
        # Dirichlet priors) and discard burn-in 
        chain = self.prior.transform(_chain)
        lnpost = zeus_sampler.get_log_prob()[burnin:,:]

        output = self._save_chains(
                chain, 
                lnpost, 
                lnpost_args, 
                lnpost_kwargs, 
                writeout=writeout,
                overwrite=overwrite, 
                debug=debug, 
                **kwargs) 
        return output 

    def _initialize_walkers(self, lnpost_args, lnpost_kwargs, 
            nwalkers=100, opt_maxiter=1000, theta_start=None, debug=False):
        ''' initalize the walkers by minimizing the -2 * lnPost
        '''
        # initialize the walkers 
        if debug: print('--- initializing the walkers ---') 
        ndim = self.prior.ndim_sampling

        x0 = np.mean([self.prior.sample() for i in range(10)], axis=0)
        std0 = np.std([self.prior.sample() for i in range(10)], axis=0)
        
        if theta_start is None: 
            # initialize optimiser 
            _lnpost = lambda *args: -2. * self.lnPost(*args, **lnpost_kwargs) 
            min_result = op.minimize(
                    _lnpost, 
                    x0, 
                    args=lnpost_args, 
                    method='Nelder-Mead', 
                    options={'maxiter': opt_maxiter}) 
            tt0 = min_result['x'] 
            logp0 = -0.5*min_result['fun']
        else: 
            tt0 = theta_start 
            if tt0.shape[0] == nwalkers: return tt0

            logp0 = self.lnPost(tt0, *lnpost_args, **lnpost_kwargs) 
        if debug:
            print('initial theta = [%s]' % ', '.join([str(_t) for _t in tt0])) 
            print('log Posterior(theta0) = %f' % logp0)
        # initial walker positions 
        p0 = [tt0 + 1e-3 * std0 * np.random.randn(ndim) for i in range(nwalkers)]
        # chekc that they're within the prior
        for i in range(nwalkers): 
            while not np.isfinite(self.lnPrior(p0[i])): 
                p0[i] = tt0 + 1e-3 * std0 * np.random.randn(ndim)
        return p0

    def _save_chains(self, chain, lnpost, lnpost_args, lnpost_kwargs, writeout=None,
            overwrite=False, debug=False, **kwargs):
        ''' save MC chains to file along with other details. If file exists, it
        will append it to the hdf5 file unless `overwrite=True`. 
        '''
        if writeout is None: 
            pass
        elif not overwrite and os.path.isfile(writeout): 
            raise NotImplementedError
            '''
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
            '''
        else:   
            if debug: print('  writing to ... %s' % writeout)
            mcmc = h5py.File(writeout, 'w')  # write 
            mcmc.create_dataset('mcmc_chain0', data=chain) # first chain 
            mcmc.create_dataset('log_prob0', data=lnpost) 
            newfile = True
    
        # get summary of the posterior from *all* of the chains in file 
        flat_chain = self._flatten_chain(chain)
        flat_log_prob = lnpost.reshape(np.prod(lnpost.shape))

        i_max = flat_log_prob.argmax() 
        theta_bestfit = flat_chain[i_max,:]

        if debug: 
            print('bestfit theta = [%s]' % ', '.join([str(_t) for _t in
                theta_bestfit])) 
            print('log Posterior = %f' % (flat_log_prob[i_max]))
    
        output = {} 
        output['theta_bestfit'] = theta_bestfit
        
        if writeout is None:
            output['mcmc_chain'] = chain 
            output['log_prob'] = lnpost 
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
        output['log_prob'] = lnpost 
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
                try: 
                    ichain = int(k.replace('mcmc_chain', ''))
                    i_chains.append(ichain)
                except ValueError: 
                    pass 
            else: 
                mcmc[k] = chains[k][...]
        mcmc['redshift'] = float(mcmc['redshift']) 

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
    def __init__(self, model=None, flux_calib=None, prior=None, corrprior=None): 
        if model is None: # default Model class object 
            from .models import NMF
            self.model = NMF(burst=False, emulator=True)
        else: 
            self.model = model

        if flux_calib is None: # default FluxCalib function  
            from .flux_calib import no_flux_factor 
            self.flux_calib = no_flux_factor
        else: 
            self.flux_calib = flux_calib

        self.prior = prior 
        assert 'sed' in self.prior.labels, 'please label which priors are for the SED'

        self.corrprior = corrprior
    
    def run(self, wave_obs=None, flux_obs=None, flux_ivar_obs=None,
            resolution=None, photo_obs=None, photo_ivar_obs=None, zred=None,
            vdisp=150., mask=None, bands=None, sampler='emcee',
            nwalkers=100, niter=1000, burnin=100, maxiter=200000,
            opt_maxiter=100, theta_start=None, writeout=None, overwrite=False, debug=False,
            progress=True, pool=None, **kwargs): 
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

        resolution : array_like
            resolution matrix in sparse matrix from. 
        
        photo_obs : array_like[Nband,]
             observed photometric flux __in units of nanomaggies__

        photo_ivar_obs : array_like[Nband,]
            observed photometric flux **inverse variance**. Not uncertainty!

        zred : float 
            redshift of the observations  

        vdisp : float
            velocity disperion in km/s. 
            (Default: 150.) 
    
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
            - output['redshift']        : redshift 
            - output['theta_bestfit']   : parameter value of MCMC sample with highest log probability 
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
                wave_obs=wave_obs, flux_obs=flux_obs,
                flux_ivar_obs=flux_ivar_obs, resolution=resolution, 
                photo_obs=photo_obs, photo_ivar_obs=photo_ivar_obs, zred=zred,
                vdisp=vdisp, mask=mask, bands=bands)

        self._lnpost_args = lnpost_args
        self._lnpost_kwargs = lnpost_kwargs
        
        # run MCMC 
        if sampler == 'emcee': 
            mcmc_sampler = self._emcee
        elif sampler == 'zeus': 
            mcmc_sampler = self._zeus

        output = mcmc_sampler( 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers,
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter, 
                theta_start=theta_start, 
                writeout=writeout, 
                overwrite=overwrite, 
                progress=progress, 
                pool=pool,
                debug=debug) 
        return output  

    def restart(self, fmcmc): 
        ''' restart mcmc chain from mcmc file 
        '''
        return None 

    def lnLike(self, tt, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, vdisp, resolution=None, mask=None,
            filters=None, obs_data_type=None, debug=False):
        ''' calculated the log likelihood. 
        '''
        # separate SED parameters from Flux Calibration parameters
        tt_sed, tt_fcalib = self.prior.separate_theta(tt, 
                labels=['sed', 'flux_calib'])
        
        # calculate SED model(theta) 
        _sed = self.model.sed(tt_sed, zred, vdisp=vdisp, 
                wavelength=wave_obs, resolution=resolution, filters=filters)
        if 'photo' in obs_data_type: _, _flux, photo = _sed
        else: _, _flux = _sed

        _chi2_spec, _chi2_photo = 0., 0.
        if 'spec' in obs_data_type: 
            # apply flux calibration model
            if self._Nbins is not None: 
                Nwaves = np.cumsum([0]+self._Nbins) 
                _flux = [_flux[Nwaves[i]:Nwaves[i+1]] for i in range(len(self._Nbins))] 
            flux = self.flux_calib(tt_fcalib, _flux)

            # data - model(theta) with masking 
            dflux = (flux[~mask] - flux_obs[~mask]) 
            if debug: print(dflux)

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

    def _lnPost_args_kwargs(self, wave_obs=None, flux_obs=None,
            flux_ivar_obs=None, resolution=None, photo_obs=None,
            photo_ivar_obs=None, zred=None, vdisp=None, mask=None,
            bands=None):
        ''' preprocess all the inputs and get arg and kwargs for lnPost method 
        '''
        # check inputs
        obs_data_type = self._obs_data_type(wave_obs, flux_obs, flux_ivar_obs,
                photo_obs, photo_ivar_obs) 

        if isinstance(wave_obs, list): 
            # if separate wavelength bins are provided in a list (e.g.
            # different arms fo the spectrograph) 
            
            # save the bin sizes for flux calibrating the fluxes separately 
            self._Nbins = [len(_w) for _w in wave_obs]

            wave_obs = np.concatenate(wave_obs) 
            flux_obs = np.concatenate(flux_obs) 
            flux_ivar_obs = np.concatenate(flux_ivar_obs) 
        else:
            self._Nbins = None 
        
        if photo_obs is not None: 
            assert bands is not None, "specify the photometric bands (e.g.  'desi')" 
        
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
                zred,
                vdisp) 
        lnpost_kwargs = {
                'resolution': resolution, # resolution data 
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
   
    def _save_chains(self, chain, lnpost, lnpost_args, lnpost_kwargs, writeout=None,
            overwrite=False, debug=False, **kwargs):
        output = super()._save_chains(chain, lnpost, lnpost_args, lnpost_kwargs,
                writeout=writeout, overwrite=overwrite, debug=debug)

        obs_data_type = lnpost_kwargs['obs_data_type']
        
        wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred, vdisp = lnpost_args

        output['redshift']              = zred
        output['wavelength_obs']        = wave_obs
        output['flux_spec_obs']         = flux_obs
        output['flux_ivar_spec_obs']    = flux_ivar_obs
        output['flux_photo_obs']        = photo_obs
        output['flux_ivar_photo_obs']   = photo_ivar_obs

        # add best-fit model to output dictionary
        tt_sed, tt_fcalib = self.prior.separate_theta(output['theta_bestfit'],
                labels=['sed', 'flux_calib'])
        
        _sed = self.model.sed(tt_sed, zred, vdisp=vdisp, wavelength=wave_obs,
                resolution=lnpost_kwargs['resolution'],
                filters=lnpost_kwargs['filters'])
        if 'photo' in obs_data_type: 
            _, _flux, photo = _sed
        else: _, _flux = _sed

        if 'photo' in obs_data_type: 
            output['flux_photo_model'] = photo
        if 'spec' in obs_data_type: 
            # apply flux calibration model
            if self._Nbins is not None: 
                Nwaves = np.cumsum([0]+self._Nbins) 
                _flux = [_flux[Nwaves[i]:Nwaves[i+1]] for i in range(len(self._Nbins))] 
            flux = self.flux_calib(tt_fcalib, _flux).flatten()
            output['flux_spec_model'] = flux 

        if writeout is not None: 
            # append the extra columns to file 
            if overwrite: 
                mcmc = h5py.File(writeout, 'w') 
            else: 
                mcmc = h5py.File(writeout, 'a') 

            for k in output.keys(): 
                if k not in mcmc.keys() and output[k] is not None: 
                    if k == 'mcmc_chain': pass 
                    mcmc.create_dataset(k, data=output[k]) 
            mcmc.close() 
        return output 
    

class nsaMCMC(desiMCMC): 
    ''' MCMC inference object specifically designed for analyzing NSA UV and
    optical photometry 
    '''
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
            if bands == 'nsa': 
                # load galex filters 
                fuv = specFilter.load_filter(
                        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'galex-fuv.ecsv'))
                nuv = specFilter.load_filter(
                        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'galex-nuv.ecsv'))
                # load sdss filters
                sdss_u = specFilter.load_filter('sdss2010-u')
                sdss_g = specFilter.load_filter('sdss2010-g')
                sdss_r = specFilter.load_filter('sdss2010-r')
                sdss_i = specFilter.load_filter('sdss2010-i')
                sdss_z = specFilter.load_filter('sdss2010-z')

                filters = [fuv, nuv, sdss_u, sdss_g, sdss_r, sdss_i, sdss_z]
            elif bands == 'sdss':
                # load sdss filters
                sdss_u = specFilter.load_filter('sdss2010-u')
                sdss_g = specFilter.load_filter('sdss2010-g')
                sdss_r = specFilter.load_filter('sdss2010-r')
                sdss_i = specFilter.load_filter('sdss2010-i')
                sdss_z = specFilter.load_filter('sdss2010-z')

                filters = [sdss_u, sdss_g, sdss_r, sdss_i, sdss_z]
            else: 
                raise NotImplementedError("specified bands not implemented") 
        else: 
            raise NotImplementedError("specified bands not implemented") 

        return specFilter.FilterSequence(filters)


# --- priors --- 
def default_NMF_prior(burst=True): 
    ''' default prior for NMF model
    '''
    prior_list = [
            UniformPrior(7., 12.5, label='sed'),
            FlatDirichletPrior(4, label='sed')   # flat dirichilet priors
            ]
    if burst: 
        prior_list.append(UniformPrior(0., 1., label='sed')) # burst fraction
        prior_list.append(LogUniformPrior(1e-2, 13.27, label='sed')) # tburst
    
    prior_list.append(LogUniformPrior(4.5e-5, 4.5e-2, label='sed')) # uniform priors on ZH coeff
    prior_list.append(LogUniformPrior(4.5e-5, 4.5e-2, label='sed')) # uniform priors on ZH coeff
    prior_list.append(UniformPrior(0., 3., label='sed'))        # uniform priors on dust1
    prior_list.append(UniformPrior(0., 3., label='sed'))        # uniform priors on dust2
    prior_list.append(UniformPrior(-2., 1, label='sed'))     # uniform priors on dust_index
    
    return load_priors(prior_list) 


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
            _lnp_theta = prior.lnPrior(theta[:,i:i+prior.ndim_sampling]) 
            if not np.isfinite(_lnp_theta): return -np.inf
            
            lnp_theta += _lnp_theta 
            i += prior.ndim_sampling

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
        tt_p = np.empty(tt.shape[:-1]+(self.ndim,)) 

        i, _i = 0, 0 
        for prior in self.list_of_priors: 
            tt_p[...,i:i+prior.ndim] = prior.transform(tt[...,_i:_i+prior.ndim_sampling])
            i += prior.ndim
            _i += prior.ndim_sampling
        return tt_p 

    def untransform(self, tt): 
        ''' transform theta 
        '''
        tt_p = np.empty(tt.shape[:-1]+(self.ndim_sampling,)) 

        i, _i = 0, 0 
        for prior in self.list_of_priors: 
            tt_p[...,i:i+prior.ndim_sampling] = prior.untransform(tt[...,_i:_i+prior.ndim])
            i += prior.ndim_sampling
            _i += prior.ndim
        return tt_p 


    def separate_theta(self, theta, labels=None): 
        ''' separate theta based on label
        '''
        lbls = np.concatenate([np.repeat(prior.label, prior.ndim) 
            for prior in self.list_of_priors]) 

        output = [] 
        for lbl in labels: 
            islbl = (lbls == lbl) 
            output.append(theta[islbl])
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
    def ndim_sampling(self): 
        # update ndim  
        return np.sum([prior.ndim_sampling for prior in self.list_of_priors]) 

    @property
    def labels(self): 
        return np.array([prior.label for prior in self.list_of_priors]) 

    @property 
    def range(self): 
        ''' range of the priors 
        '''
        prior_min, prior_max = [], [] 
        for prior in self.list_of_priors: 
            if isinstance(prior, UniformPrior): 
                _min = prior.min
                _max = prior.max 
            elif isinstance(prior, LogUniformPrior): 
                _min = prior.min
                _max = prior.max 
            elif isinstance(prior, FlatDirichletPrior): 
                _min = np.zeros(prior.ndim) 
                _max = np.ones(prior.ndim) 
            elif isinstance(prior, GaussianPrior): 
                _min = prior.mean - 3.* np.sqrt(np.diag(prior.covariance)) 
                _max = prior.mean + 3.* np.sqrt(np.diag(prior.covariance)) 
            else: 
                raise ValueError
            prior_min.append(np.atleast_1d(_min))
            prior_max.append(np.atleast_1d(_max)) 
        prior_min = np.concatenate(prior_min) 
        prior_max = np.concatenate(prior_max) 
        return prior_min, prior_max


class Prior(object): 
    ''' base class for prior
    '''
    def __init__(self, label=None):
        self.ndim = None 
        self.ndim_sampling = None 
        self.label = label 
        self._random = np.random.mtrand.RandomState()

    def transform(self, tt): 
        ''' Some priors require transforming the parameter space for sampling
        (e.g. Dirichlet priors). For most priors, this transformation will return 
        the same value  
        '''
        return tt 
    
    def untransform(self, tt): 
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
        self.ndim_sampling = ndim - 1

    def transform(self, tt): 
        ''' warped manifold transformation as specified in Betancourt (2013).
        This function transforms samples from a uniform distribution to a
        Dirichlet distribution .

        x_i = (\prod\limits_{k=1}^{i-1} z_k) * f 

        f = 1 - z_i         for i < m
        f = 1               for i = m 
    
        Parameters
        ----------
        tt : array_like[N,m-1]
            N samples drawn from a (m-1)-dimensional uniform distribution 

        Returns
        -------
        tt_d : array_like[N,m]
            N transformed samples drawn from a m-dimensional dirichlet
            distribution 

        Reference
        ---------
        * Betancourt(2013) - https://arxiv.org/pdf/1010.3436.pdf
        '''
        assert tt.shape[-1] == self.ndim_sampling
        tt_d = np.empty(tt.shape[:-1]+(self.ndim,)) 
    
        tt_d[...,0] = 1. - tt[...,0]
        for i in range(1,self.ndim_sampling): 
            tt_d[...,i] = np.prod(tt[...,:i], axis=-1) * (1. - tt[...,i]) 
        tt_d[...,-1] = np.prod(tt, axis=-1) 
        return tt_d 

    def untransform(self, tt_d): 
        ''' reverse the warped manifold transformation 
        '''
        assert tt_d.shape[-1] == self.ndim 
        tt = np.empty(tt_d.shape[:-1]+(self.ndim_sampling,)) 

        tt[...,0] = 1. - tt_d[...,0]
        for i in range(1,self.ndim_sampling): 
            tt[...,i] = 1. - (tt_d[...,i]/np.prod(tt[...,:i], axis=-1))
        return tt

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
        if np.all(theta <= np.ones(self.ndim_sampling)) and np.all(theta >= np.zeros(self.ndim_sampling)): 
            return 0.
        else: 
            return -np.inf
    
    def append(self, *arg, **kwargs): 
        raise ValueError("appending not supproted") 

    def sample(self): 
        return np.array([self._random.uniform(0., 1.) for i in range(self.ndim_sampling)])


class UniformPrior(Prior): 
    ''' uniform tophat prior
    
    '''
    def __init__(self, _min, _max, label=None):
        super().__init__(label=label)
        
        self.min = np.atleast_1d(_min)
        self.max = np.atleast_1d(_max)
        self.ndim = self.min.shape[0]
        self.ndim_sampling = self.ndim
        assert self.min.shape[0] == self.max.shape[0]
        assert np.all(self.min <= self.max)
        
    def lnPrior(self, theta):
        if np.all(theta <= self.max) and np.all(theta >= self.min): 
            return 0.
        else:
            return -np.inf

    def sample(self): 
        return np.array([self._random.uniform(mi, ma) for (mi, ma) in zip(self.min, self.max)])


class LogUniformPrior(Prior): 
    ''' log uniform tophat prior
    
    '''
    def __init__(self, _min, _max, label=None):
        super().__init__(label=label)
        
        self.min = np.atleast_1d(_min)
        self.max = np.atleast_1d(_max)
        self.ndim = self.min.shape[0]
        self.ndim_sampling = self.ndim
        assert self.min.shape[0] == self.max.shape[0]
        assert np.all(self.min <= self.max)
        
    def lnPrior(self, theta):
        if np.all(theta <= self.max) and np.all(theta >= self.min): 
            return 0.
        else:
            return -np.inf

    def sample(self): 
        return np.array([10**self._random.uniform(np.log10(mi), np.log10(ma)) for mi, ma in zip(self.min, self.max)])


class GaussianPrior(Prior): 
    ''' Gaussian prior 
    '''
    def __init__(self, mean, covariance, label=None):
        super().__init__(label=label)
        
        self.mean = np.atleast_1d(mean)
        self.covariance = np.atleast_1d(covariance)
        self.ndim = self.mean.shape[0]
        self.ndim_sampling = self.ndim
        assert self.mean.shape[0] == self.covariance.shape[0]
        self.multinorm = stat.multivariate_normal(self.mean, self.covariance)
    
    def lnPrior(self, theta):
        return self.multinorm.logpdf(theta) 

    def sample(self): 
        return np.atleast_1d(self.multinorm.rvs())


class PostOut(object): 
    ''' posterior output object
    '''
    def __init__(self): 
        self._samples   = None # mcmc samples 
        self._log_prob  = None # log probabilities 
        self._theta_bf  = None # best-fit theta 
    
        self._redshift  = None # observed redshift
        self._w_obs     = None # wavelength of observed spectra 
        self._flux_spec_obs     = None # flux of observed spectra 
        self._ivar_spec_obs     = None # inverse variance of observed spectra 
        self._flux_photo_obs    = None # flux of observed photometry 
        self._ivar_photo_obs    = None # inverse variance of observed photometry 

        self._flux_spec_model   = None # flux of model spectra 
        self._ivar_spec_model   = None # inverse variance of model spectra 
        self._flux_photo_model  = None # flux of model photometry 
        self._ivar_photo_model  = None # inverse variance of model photometry 

    def read(self, fname): 
        ''' read posterior from file 
        '''
        mcmc = h5py.File(fname, 'r')  # write 
        
        if 'samples' in mcmc.keys(): 
            self.samples    = mcmc['samples'][...]
        elif 'mcmc_chain' in mcmc.keys(): 
            self.samples    = mcmc['mcmc_chain'][...]
        else: 
            raise ValueError
        self.log_prob   = mcmc['log_prob'][...]
        
        if 'redshift' in mcmc.keys(): 
            self.redshift = mcmc['redshift'][...]
        if 'wavelength_obs' in mcmc.keys(): 
            self.wavelength_obs = mcmc['wavelength_obs'][...]
        if 'flux_spec_obs' in mcmc.keys(): 
            self.flux_spec_obs = mcmc['flux_spec_obs'][...]
        if 'ivar_spec_obs' in mcmc.keys(): 
            self.ivar_spec_obs = mcmc['ivar_spec_obs'][...]
        if 'flux_photo_obs' in mcmc.keys(): 
            self.flux_photo_obs = mcmc['flux_photo_obs'][...]
        if 'ivar_photo_obs' in mcmc.keys(): 
            self.ivar_photo_obs = mcmc['ivar_photo_obs'][...]
    
        # to be backwards compatible. 
        if 'flux_ivar_spec_obs' in mcmc.keys(): 
            self.ivar_spec_obs = mcmc['flux_ivar_spec_obs'][...]
        if 'flux_ivar_photo_obs' in mcmc.keys(): 
            self.ivar_photo_obs = mcmc['flux_ivar_photo_obs'][...]


        if 'flux_spec_model' in mcmc.keys(): 
            self.flux_spec_model = mcmc['flux_spec_model'][...]
        if 'flux_photo_model' in mcmc.keys(): 
            self.flux_photo_model = mcmc['flux_photo_model'][...]

        mcmc.close() 
        return None 
    
    def write(self, fname, overwrite=True): 
        ''' write posterior to given file. 
    
        Parameters
        ----------
        fname : str
            hdf5 file name to write the posterior to

        overwrite : bool
            If True, overwrite file. If False, return error 
        '''
        if os.path.isfile(fname) and not overwrite: 
            raise ValueError

        mcmc = h5py.File(fname, 'w')  # write 
    
        assert self.samples is not None
        assert self.log_prob is not None 

        mcmc.create_dataset('samples', self.samples)
        mcmc.create_dataset('log_prob', self.log_prob)

        mcmc.create_dataset('theta_bestfit', self.theta_bestfit)
        
        # if observations are provided
        if self.redshift is not None:
            mcmc.create_dataset('redshift', self.redshift) 
        if self.wavelength_obs is not None:
            mcmc.create_dataset('wavelength_obs', self.wavelength_obs)
        if self.flux_spec_obs is not None:
            mcmc.create_dataset('flux_spec_obs', self.flux_spec_obs)
        if self.ivar_spec_obs is not None:
            mcmc.create_dataset('ivar_spec_obs', self.ivar_spec_obs)
        if self.flux_photo_obs is not None:
            mcmc.create_dataset('flux_photo_obs', self.flux_photo_obs)
        if self.ivar_photo_obs is not None:
            mcmc.create_dataset('ivar_photo_obs', self.ivar_photo_obs)

        # if best-fit model are provided
        if self.flux_spec_model is not None: 
            mcmc.create_dataset('flux_spec_model', self.flux_spec_model)
        if self.flux_photo_model is not None: 
            mcmc.create_dataset('flux_photo_model', self.flux_photo_model)
        mcmc.close() 
        return None 

    def validate(self, nburnin=500, labels=None, plot_range=None): 
        ''' validation plot of the posterior
        '''
        import matplotlib.pyplot as plt
        import corner as DFM
        
        flat_chain = self.flatten_chain(self.samples[nburnin:,:,:])
        ndim = flat_chain.shape[-1]

        if labels is None: 
            # default labels for fititng spectrophotometry
            lbls_default = [r'$\log M_*$', 
                    r'$\beta^{\rm SFH}_1$', r'$\beta^{\rm SFH}_2$', 
                    r'$\beta^{\rm SFH}_3$', r'$\beta^{\rm SFH}_4$',
                    r'$f_{\rm burst}$', r'$t_{\rm burst}$', 
                    r'$\gamma_1^{\rm ZH}$', r'$\gamma_2^{\rm ZH}$',
                    r'$\tau_{\rm BC}$', r'$\tau_{\rm ISM}$', r'$n_{\rm dust}$', 
                    r'$f_{\rm fiber}$']
            if len(lbls_default) == ndim:
                labels = lbls_default 

        if plot_range is None: 
            # default range for fititng spectrophotometry
            range_default = [(7, 12.5), 
                    (0., 1.), (0., 1.), (0., 1.), (0., 1.), 
                    (0., 1.), (1e-2, 13.27), 
                    (4.5e-5, 1.5e-2), (4.5e-5, 1.5e-2), 
                    (0., 3.), (0., 3.), (-2., 1.), 
                    (0., 1.)]
            if len(range_default) == ndim: 
                plot_range = range_default 

        fig = plt.figure(figsize=(15, 20))
        gs0 = fig.add_gridspec(nrows=ndim, ncols=ndim, top=0.95, bottom=0.275)
        for yi in range(ndim):
            for xi in range(ndim):
                sub = fig.add_subplot(gs0[yi, xi])

        _fig = DFM.corner(
                flat_chain[::10,:],
                quantiles=None,
                levels=[0.68, 0.95],
                bins=20,
                smooth=True,
                labels=labels,
                label_kwargs={'fontsize': 20, 'labelpad': 0.1},
                range=plot_range,
                fig=fig)

        axes = np.array(fig.axes).reshape((ndim, ndim))
        for yi in range(1, ndim):
            ax = axes[yi, 0]
            ax.set_ylabel(labels[yi], fontsize=20, labelpad=30)
            ax.yaxis.set_label_coords(-0.6, 0.5)
        for xi in range(ndim):
            ax = axes[-1, xi]
            ax.set_xlabel(labels[xi], fontsize=20, labelpad=30)
            ax.xaxis.set_label_coords(0.5, -0.55)

            ax = axes[xi, xi]
            ax.set_xlim(plot_range[xi])

        gs1 = fig.add_gridspec(nrows=1, ncols=30, top=0.2, bottom=0.05)
        sub = fig.add_subplot(gs1[0, :7])
        
        if self.flux_photo_obs is not None: 
            mags_obs = 22.5 - 2.5 * np.log10(self.flux_photo_obs)
            mags_sig_obs = np.abs(-2.5 *
                    (self.ivar_photo_obs**-0.5)/self.flux_photo_obs/np.log(10))
            sub.errorbar([4720., 6415., 9260.], mags_obs,
                    yerr=mags_sig_obs, fmt='.k')
        if self.flux_photo_model is not None: 
            mags_model = 22.5 - 2.5 * np.log10(self.flux_photo_model)
            sub.scatter([4720., 6415., 9260.], mags_model, marker='s',
                    facecolor='none', s=70, c='C1')
        sub.set_xlim(4000, 1e4)
        sub.set_xticks([4720., 6415., 9260])
        sub.set_xticklabels(['g', 'r', 'z'], fontsize=25)
        sub.set_ylabel('magnitude', fontsize=25)

        sub = fig.add_subplot(gs1[0, 10:])
        if self.wavelength_obs is not None and self.flux_spec_obs is not None: 
            sub.plot(self.wavelength_obs, self.flux_spec_obs, c='k', lw=0.5,
                    label='Obs.')
        if self.wavelength_obs is not None and self.flux_spec_model is not None: 
            sub.plot(self.wavelength_obs, self.flux_spec_model, c='C1', lw=1,
                    label='best-fit model')
        sub.legend(loc='upper right', fontsize=20, handletextpad=0.2)
        sub.set_xlabel('wavelength [$A$]', fontsize=25)
        sub.set_xlim(3.6e3, 9.8e3)
        sub.set_ylim(0., 20)
        sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=20)
        return fig 

    def flatten_chain(self, chain): 
        ''' flatten mcmc . If chain object is 2D then it assumes it's
        already flattened. 
        '''
        if len(chain.shape) == 2: return chain # already flat 

        self._n_iter    = chain.shape[0]
        self._n_walker  = chain.shape[1]
        self._n_param   = chain.shape[2]

        s = list(chain.shape[1:])
        s[0] = np.prod(chain.shape[:2]) 
        return chain.reshape(s)

    @property
    def samples(self) : 
        return self._samples 

    @samples.setter
    def samples(self, samples): 
        self._samples = samples

    @property
    def log_prob(self) : 
        return self._log_prob

    @log_prob.setter
    def log_prob(self, log_prob): 
        self._log_prob = log_prob

    @property
    def theta_bestfit(self): 
        assert self._log_prob is not None
        assert self._samples is not None
        
        flat_chain = self.flatten_chain(self.samples) 
        flat_log_prob = self.log_prob.reshape(np.prod(self.log_prob.shape))

        i_max = flat_log_prob.argmax() 
        self._theta_bf = flat_chain[i_max,:]
        return self._theta_bf

    @property 
    def redshift(self): 
        return self._redshift

    @redshift.setter 
    def redshift(self, zred): 
        self._redshift = zred
        return self._redshift

    @property 
    def wavelength_obs(self): 
        return self._w_obs 

    @wavelength_obs.setter 
    def wavelength_obs(self, wave): 
        self._w_obs = wave
        return self._w_obs 
    
    @property 
    def flux_spec_obs(self): 
        return self._flux_spec_obs

    @flux_spec_obs.setter
    def flux_spec_obs(self, flux): 
        self._flux_spec_obs = flux 
        return self._flux_spec_obs

    @property 
    def ivar_spec_obs(self): 
        return self._ivar_spec_obs

    @ivar_spec_obs.setter
    def ivar_spec_obs(self, ivar): 
        self._ivar_spec_obs = ivar
        return self._ivar_spec_obs
    
    @property 
    def flux_photo_obs(self): 
        return self._flux_photo_obs

    @flux_photo_obs.setter
    def flux_photo_obs(self, flux): 
        self._flux_photo_obs = flux 
        return self._flux_photo_obs

    @property 
    def ivar_photo_obs(self): 
        return self._ivar_photo_obs

    @ivar_photo_obs.setter
    def ivar_photo_obs(self, ivar): 
        self._ivar_photo_obs = ivar
        return self._ivar_photo_obs

    @property 
    def flux_spec_model(self): 
        return self._flux_spec_model

    @flux_spec_model.setter
    def flux_spec_model(self, flux): 
        self._flux_spec_model = flux 
        return self._flux_spec_model
    
    @property 
    def flux_photo_model(self): 
        return self._flux_photo_model

    @flux_photo_model.setter
    def flux_photo_model(self, flux): 
        self._flux_photo_model = flux 
        return self._flux_photo_model
