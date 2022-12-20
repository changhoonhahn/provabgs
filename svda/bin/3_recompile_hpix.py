'''


recompile healpix posterior files with updated posteriors 


'''
import os, sys
import h5py
import numpy as np 

from provabgs import infer as Infer

#######################################################
# inputs
#######################################################
target = sys.argv[1]
survey = sys.argv[2]

#######################################################

# get healpixels and target ids of rerun posteriors 
dat_dir = '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/'

hpixs, tids = np.loadtxt(os.path.join(dat_dir, f'{survey}-bright-{target}.flagged.dat'), 
                         dtype=int, unpack=True, usecols=[0,1])
uhpixs = np.unique(hpixs)

missed_hpix, missed_tids = [], [] 
for hpix in uhpixs: 
    tids_redone = tids[hpixs == hpix]
        
    print('updating %i posteriors in healpix %i' % (len(tids_redone), hpix))

    # read hpix file 
    with h5py.File(os.path.join(dat_dir, f'provabgs-{survey}-bright-{hpix}.{target}.hdf5'), 'r+') as fhpix:
        for tid in tids_redone: 
            is_tid = (fhpix['targetid'][...] == tid)
            if np.sum(is_tid) == 0: continue # hmm, it's not in the original file for whatever reason 
            i_tid = np.arange(len(fhpix['targetid'][...]))[is_tid][0]

            # read updated posterior file  
            fmcmc = os.path.join(dat_dir, 'healpix', str(hpix), 'provabgs.%i.hdf5' % tid)

            if not os.path.isfile(fmcmc): 
                print(fmcmc) 
                print('\t healpix %i targetid %i missing' % (hpix, tid))
                missed_hpix.append(hpix)
                missed_tids.append(tid) 
                continue 

            post_i = Infer.PostOut()
            post_i.read(fmcmc)
    
            # update value 
            fhpix['samples'][i_tid]     = post_i.samples 
            fhpix['log_prob'][i_tid]    = post_i.log_prob
            fhpix['redshift'][i_tid]    = post_i.redshift
            fhpix['wavelength_obs'][i_tid]  = post_i.wavelength_obs
            fhpix['flux_spec_obs'][i_tid]   = post_i.flux_spec_obs
            fhpix['ivar_spec_obs'][i_tid]   = post_i.ivar_spec_obs
            fhpix['flux_photo_obs'][i_tid]  = post_i.flux_photo_obs
            fhpix['ivar_photo_obs'][i_tid]  = post_i.ivar_photo_obs

            fhpix['flux_spec_model'][i_tid]   = post_i.flux_spec_model
            fhpix['flux_photo_model'][i_tid]  = post_i.flux_spec_model # this is a bug but it will blow up otherwise

np.savetxt(os.path.join(dat_dir, f'{survey}-bright-{target}.missed.dat'),
           np.array([missed_hpix, missed_tids]).T, fmt='%i %i')
