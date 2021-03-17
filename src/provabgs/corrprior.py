'''

module for imposing prior correction 

'''
import numpy as np 


class CorrectPrior(object): 
    ''' CorrectPrior object can be used to impose uniform priors on multiple 
    derived galaxy properties using maximum entropy prior correction. 
    
    ***usage examples here***
    
    Parameters
    ----------
    model : models.Model object
        SPS model object that is used to calculate derived galaxy properties
        (e.g. average SFR, metallicity) 

    prior : infer.Prior or infer.PriorSeq object
        prior object that's used to sample parameters from the original prior.  

    tage : float, optional 
        age of the galaxy in Gyrs. Either tage or zred must be specified. 
        
    zred : float, optional
        redshift of the galaxy. Either tage or zred must be specified. 

    prop : list of strings
        specifies the galaxy properties to impose uniform priors on. Currently
        the following properties are supported: 
        * ['logmstar', 'avgsfr_1gyr', 'logavgsfr_1gyr', 'z_mw']

    Nprior : int
        Number of samples to draw from the prior in order to estimate the PDF
        of prior on derived properties. More the better. 

    method : string
        method for the density estimation. Currently only 'kde' and 'gmm'
        supported. 

    debug : boolean
        If True, prints things for debugging purposes. 


    Notes 
    -----
    *
    '''
    def __init__(self, model, prior, tage=None, zred=None, props=['logmstar',
        'avgsfr_1gyr', 'z_mw'], Nprior=100000, method='kde', debug=False,
        **method_kwargs):

        self.model = model
        self.prior = prior 

        self.tage = tage
        self.zred = zred 
        self.props = props 
        self.Nprior = Nprior  
        self.method = method

        # check that properties are named 
        self.supported_props = ['logmstar', 'avgsfr_1gyr', 'logavgsfr_1gyr',
                'avgssfr_1gyr', 'logavgssfr_1gyr', 'z_mw']
        for prop in self.props:
            assert prop in self.supported_props, (
                    "%s not supported \ncurrently %s are supported" % 
                    (prop, ', '.join(self.supported_props)))

        # fit the prior on specified properties
        self._fit_prior(debug=debug, **method_kwargs)

    def get_importance_weights(self, theta, outlier=0.01, debug=False): 
        ''' calculate importance weights for given parameter values. The input
        parameters should be the parameter values from the MCMC chain 

        
        Parameters
        ----------
        theta : 2d array
            N x Ndim array of parameter values

        outlier : float 
             
        '''
        ftheta = self._get_properties(theta) 
    
        p_ftheta = self.p_ftheta.pdf(ftheta)

        # clip probabilities that are on the edges 
        low_lim, high_lim = np.percentile(p_ftheta, 
                [0.5*outlier, 100.-0.5*outlier]) 
        if debug: print('... clipping values outside %.5e, %.5e' % (low_lim, high_lim))

        return 1./np.clip(p_ftheta, low_lim, high_lim) 

    def _clip_prob(self, probs): 
        '''
        '''

    def _fit_prior(self, debug=False, **method_kwargs): 
        ''' fit the prior distribution of derived properties.
        '''
        # sample Nprior thetas from prior 
        _theta_prior = np.array([self.prior.sample() for i in range(self.Nprior)])
        theta_prior = self.prior.transform(_theta_prior)
        
        # get specified properties from theta and model
        ftheta = self._get_properties(theta_prior, debug=debug) 
        
        # fit PDF of derived properties
        if debug: print('... fitting prior(derived prop)') 
        self.p_ftheta = _fit_pdf(ftheta, method=self.method, debug=debug, **method_kwargs)
        return None 

    def _get_properties(self, theta, debug=False): 
        ''' get derived properties for given parameter value
        '''
        # calculate derived properties 
        derived_props = [] 
        if 'logmstar' in self.props: # log M*
            if debug: print('... calculating log M*') 
            tt = self.model._parse_theta(theta)
            derived_props.append(tt['logmstar'])

        if 'avgsfr_1gyr' in self.props: # average SFR in the last 1 Gyr year
            assert 'logavgsfr_1gyr' not in self.props, "impossible to impose uniform priors on x and log(x) simultaneously..."
            if debug: print('... calculating avg SFR_1Gyr') 
            avgsfr = self.model.avgSFR(theta, tage=self.tage, zred=self.zred, dt=1.)
            derived_props.append(avgsfr)
        
        if 'logavgsfr_1gyr' in self.props: # avg logSFR in the last 1 Gyr year
            assert 'avgsfr_1gyr' not in self.props, "impossible to impose uniform priors on x and log(x) simultaneously..."
            if debug: print('... calculating log avg SFR_1Gyr') 
            avgsfr = self.model.avgSFR(theta, tage=self.tage, zred=self.zred, dt=1.)
            derived_props.append(np.log10(avgsfr))

        if 'avgssfr_1gyr' in self.props: # average sSFR in the last 1 Gyr year
            assert 'logavgssfr_1gyr' not in self.props, "impossible to impose uniform priors on x and log(x) simultaneously..."
            if debug: print('... calculating avg sSFR_1Gyr') 
            tt = self.model._parse_theta(theta)
            avgsfr = self.model.avgSFR(theta, tage=self.tage, zred=self.zred, dt=1.)
            derived_props.append(avgsfr/10**tt['logmstar'])

        if 'logavgssfr_1gyr' in self.props: # average log sSFR in the last 1 Gyr year
            assert 'avgssfr_1gyr' not in self.props, "impossible to impose uniform priors on x and log(x) simultaneously..."
            if debug: print('... calculating log avg sSFR_1Gyr') 
            tt = self.model._parse_theta(theta)
            avgsfr = self.model.avgSFR(theta, tage=self.tage, zred=self.zred, dt=1.)
            derived_props.append(np.log10(avgsfr) - tt['logmstar'])

        if 'z_mw' in self.props: # mass-weighted metallicity 
            if debug: print('... calculating mass-weighted Z') 
            zmw = self.model.Z_MW(theta, tage=self.tage, zred=self.zred) 
            derived_props.append(zmw) 

        return np.array(derived_props).T 


def _fit_pdf(samples, method='kde', debug=False, **method_kwargs): 
    ''' fit a probability distribution sampled by given samples using method
    specified by `method`. This function is designed to fit p(F(theta)), the
    probability distribution of *derived* properties; however, it works for any
    PDF really. 


    Parameters
    ----------
    samples : 2d array
        Nsample x Ndim array of samples from the PDF 

    method : string
        which method to use for estimating the PDF. Currently supports 'kde' and
        'gmm' (default: 'kde') 
    '''
    if debug: print('... fitting pdf using %s' % method) 
    if method == 'kde': 
        # fit PDF using Kernel Density Estimation 
        from scipy.stats import gaussian_kde as gkde

        pdf_fit = gkde(samples.T, **method_kwargs)
    else: 
        from sklearn.mixture import GaussianMixture as GMix

        if 'n_comp' not in method_kwargs.keys(): 
            raise ValueError("specify number of Gaussians `n_comp` in kwargs") 

        gmm = GMix(n_components=method_kwargs['n_comp'])
        gmm.fit(samples)
        pdf_fit = gmm
        
    return _PDF(pdf_fit, method=method) 


class _PDF(object): 
    ''' wrapper for different PDF fit objects in order to evaluate the pdf at
    and sample it conveniently.
    '''
    def __init__(self, pdf_fit, method='kde'): 
        self.pdf_fit = pdf_fit 
        self.method = method 

    def pdf(self, X): 
        if self.method == 'kde': 
            return self.pdf_fit.pdf(X.T)
        elif self.method == 'gmm': 
            return np.exp(self.pdf_fit.score_samples(X))

    def sample(self, Nsample): 
        if self.method == 'kde': 
            return self.pdf_fit.resample(Nsample)
        elif self.method == 'gmm': 
            return np.exp(self.pdf_fit.sample(Nsample))
