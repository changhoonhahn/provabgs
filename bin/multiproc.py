'''

scripts to test provabgs with multiprocessing 

'''
import time 
import numpy as np 
from provabgs import infer as Infer
from provabgs import models as Models
from multiprocessing import Pool


def multiprocessing_zeus():
    '''
    '''
    #
    fsps_emulator = Models.DESIspeculator()

    # set prior 
    priors = Infer.load_priors([
        Infer.UniformPrior(10., 10.5, label='sed'),
        Infer.FlatDirichletPrior(4, label='sed'), 
        Infer.UniformPrior(np.array([6.9e-5, 6.9e-5, 0., 0., -2.2]), np.array([7.3e-3, 7.3e-3, 3., 4., 0.4]), label='sed')
    ])
    random_theta = priors.sample() 
    wave, flux = fsps_emulator.sed(priors.transform(random_theta), 0.1)
    

    desi_mcmc = Infer.desiMCMC(prior=priors)
    t0 = time.time()
    mcmc = desi_mcmc.run(
            wave_obs=wave[0],
            flux_obs=flux[0],
            flux_ivar_obs=np.ones(flux.shape[1]),
            zred=0.1,
            sampler='zeus',
            nwalkers=20,
            burnin=10,
            opt_maxiter=1000,
            niter=100,
            pool=None, 
            debug=True)
    print()
    print('running on series takes %.f' % (time.time() - t0))
    print()

    
    import zeus 
    import multiprocessing 
    ncpu = multiprocessing.cpu_count() 
    print('%i cpus' % ncpu)

    t0 = time.time()
    lnpost_args, lnpost_kwargs = desi_mcmc._lnPost_args_kwargs(
            wave_obs=wave[0],
            flux_obs=flux[0],
            flux_ivar_obs=np.ones(flux.shape[1]),
            zred=0.1)
    start = desi_mcmc._initialize_walkers(lnpost_args, lnpost_kwargs, priors,
            nwalkers=20, opt_maxiter=1000, debug=True)

    print('--- burn-in ---') 
    pewl = Pool(processes=ncpu)
    with pewl as pool: 
        zeus_sampler = zeus.EnsembleSampler(
                desi_mcmc.nwalkers,
                desi_mcmc.prior.ndim, 
                desi_mcmc.lnPost, 
                pool=pool,
                args=lnpost_args, 
                kwargs=lnpost_kwargs)
        zeus_sampler.run_mcmc(start, 10)
    burnin = zeus_sampler.get_chain()

    print('--- running main MCMC ---') 
    pewl = Pool(processes=ncpu)
    with pewl as pool: 
        zeus_sampler = zeus.EnsembleSampler(
                desi_mcmc.nwalkers,
                desi_mcmc.prior.ndim, 
                desi_mcmc.lnPost, 
                pool=pool, 
                args=lnpost_args, 
                kwargs=lnpost_kwargs)
        zeus_sampler.run_mcmc(burnin[-1], 100)
    _chain = zeus_sampler.get_chain()
    print()
    print('running on parallel takes %.f' % (time.time() - t0))
    print()
    return None 


if __name__=="__main__":
    multiprocessing_zeus()
