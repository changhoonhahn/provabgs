import os, sys 
import h5py
import glob 
import datetime 
import numpy as np 
from astropy import table as aTable

import svda as SVDA 


def gather_healpix_posteriors(hpix, target, survey, user, niter=3000): 
    ''' compile all the posteriors for a healpix  
    '''
    from provabgs import infer as Infer
    dat_dir = '/global/cscratch1/sd/%s/provabgs/svda/healpix/' % user

    meta, _, _, _, _, _, _, _, _ = \
            SVDA.healpix(hpix, target=target, redux='fuji', survey=survey)
    ngals = len(meta)
    
    if ngals == 0: return None  

    for igal in range(ngals): 
        fmcmc = os.path.join(dat_dir, str(hpix), 
                'provabgs.%i.hdf5' % meta['TARGETID'][igal])
        if not os.path.isfile(fmcmc): 
            print('  %i of %i done' % (igal, ngals))
            return True 

        post_i = Infer.PostOut()
        post_i.read(fmcmc)

        if igal == 0: 
            targetid    = []
            samples     = [] 
            log_prob    = [] 
            redshift    = []

            wavelength_obs  = [] 
            flux_spec_obs   = [] 
            ivar_spec_obs   = []
            flux_photo_obs  = []
            ivar_photo_obs  = []

            flux_spec_model     = []
            flux_photo_model    = [] 
        else: 
            targetid.append(meta['TARGETID'][igal]) 
            samples.append(post_i.samples)
            log_prob.append(post_i.log_prob)
            redshift.append(post_i.redshift)

            wavelength_obs.append(post_i.wavelength_obs)
            flux_spec_obs.append(post_i.flux_spec_obs)
            ivar_spec_obs.append(post_i.ivar_spec_obs)
            flux_photo_obs.append(post_i.flux_photo_obs)
            ivar_photo_obs.append(post_i.ivar_photo_obs)

            flux_spec_model.append(post_i.flux_spec_model)
            flux_photo_model.append(post_i.flux_spec_model)

    fpetal = os.path.join(
            '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/'
            'provabgs-%s-bright-%i.%s.hdf5' % (survey, hpix, target)) 
    petal = h5py.File(fpetal, 'w')
    petal.create_dataset('targetid', data=np.array(targetid))
    petal.create_dataset('samples', data=np.array(samples))
    petal.create_dataset('log_prob', data=np.array(log_prob))
    petal.create_dataset('redshift', data=np.array(redshift))

    petal.create_dataset('wavelength_obs', data=np.array(wavelength_obs))
    petal.create_dataset('flux_spec_obs', data=np.array(flux_spec_obs))
    petal.create_dataset('ivar_spec_obs', data=np.array(ivar_spec_obs))
    petal.create_dataset('flux_photo_obs', data=np.array(flux_photo_obs))
    petal.create_dataset('ivar_photo_obs', data=np.array(ivar_photo_obs))

    petal.create_dataset('flux_spec_model', data=np.array(flux_spec_model))
    petal.create_dataset('flux_photo_model', data=np.array(flux_photo_model))
    petal.close() 
    return None 


users = ['chahah', 'shuang89', 'csaulder', 'msiudek']

for user in users:
    print(user) 
    dat_dir = '/global/cscratch1/sd/%s/provabgs/svda/healpix' % user
    dir_hpix = [int(os.path.basename(subdir)) for subdir in glob.glob(os.path.join(dat_dir, '*'))]
    for hpix in dir_hpix:
        print(user, hpix)
        fcomb = ('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/provabgs-sv3-bright-%i.BGS_BRIGHT.hdf5' % hpix)

        if os.path.isfile(fcomb): 
            tmod = datetime.datetime.fromtimestamp(os.path.getmtime(fcomb))
            if tmod > datetime.datetime(2022, 6, 30, 0, 0, 0, 0): 
                continue 
        #if not os.path.isfile('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/provabgs-sv3-bright-%i.BGS_BRIGHT.hdf5' % hpix):
        try:
            _ = gather_healpix_posteriors(hpix, 'BGS_BRIGHT', 'sv3', user, niter=3000)
        except UnboundLocalError:
            continue
        except ValueError:
            print("problem with this hpix")
        except: 
            print()
            print('%i weird' % hpix) 
            print()
