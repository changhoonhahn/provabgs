'''


check posteriors of healpix and check for issues with goodness-of-fit or
problematic runs 



'''
import os, sys
import numpy as np

from provabgs import util as UT
from provabgs import infer as Infer
from provabgs import models as Models


####################################################
# inputs
####################################################
target = sys.argv[1]
survey = sys.argv[2]

####################################################
# gather healpixels  
####################################################
dir_dat = '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/'

hpixs = np.array([int(f.split('-')[-1].split('.')[0]) 
                  for f in glob.glob(os.path.join(dir_dat, 
                                                  f'provabgs-{survey}-bright-*.{target}.hdf5'))])
print('%i healpixels' % len(hpixs))

####################################################
# loop through hpixels 
####################################################
flagged_hpix, flagged_targid, rchi2s = [], [], [] 

w_lines = UT.wavelength_emlines 

for hpix in hpixs:
    try:
        f = h5py.File(os.path.join(dir_dat, f'provabgs-{survey}-bright-{hpix}.{target}.hdf5'), 'r')

        targetids = f['targetid'][...].astype(int)
        redshift  = f['redshift'][...]
        w_obs     = f['wavelength_obs'][...]
        spec_obs  = f['flux_spec_obs'][...]
        spec_mod  = f['flux_spec_model'][...]
        ivar_obs  = f['ivar_spec_obs'][...]

        mcmc = f['samples'][:,:,:,0]
    except:
        print()
        print('problem opening healpix %i' % hpix)
        print()
        continue

    for igal in range(spec_obs.shape[0]):
        flagged = False

        ##################################################################
        # check goodness of fit
        ##################################################################
        # calculate chi2 to flag based on goodness of fit
        w_lines_z = w_lines * (1 + redshift[igal])

        # get mask
        mask = np.zeros(w_obs.shape[1]).astype(bool)
        # mask 40A around emission lines
        for wl in w_lines_z:
            mask = mask | ((w_obs[igal] > wl - 20) & (w_obs[igal] < wl + 20))
        # mask regions where sky subtraction is not great
        mask = mask | (w_obs[igal] > 7500)

        # chi2
        chi2 = (spec_mod[igal][~mask] - spec_obs[igal][~mask])**2 * ivar_obs[igal][~mask]
        # sigma clip
        chi2_lim = np.quantile(chi2, 0.99)

        rchi2 = np.mean(chi2[chi2 < chi2_lim])
        rchi2s.append(rchi2)

        if rchi2 > 5: # arbitrary threshold
            flagged = True
            print(rchi2, np.mean(ivar_obs[igal][~mask]))

        ##################################################################
        # check whether M* posterior hits prior limits
        ##################################################################
        mcmc_i = mcmc[igal].flatten()

        m_low, m_med, m_high = np.quantile(mcmc_i, (0.01, 0.5, 0.99))

        if (m_low < 7.1) or (m_high > 12.4):
            flagged = True
            print(m_low, m_med, m_high)
            #fig = plt.figure(figsize=(4, 4))
            #sub = fig.add_subplot(111)
            #sub.hist(mcmc_i, range=(7., 12.5), bins=40)
            #sub.set_xlim(7., 12.5)
            #plt.show()

        if flagged:
            flagged_hpix.append(hpix)
            flagged_targid.append(targetids[igal])

np.savetxt(os.path.join(dat_dir, f'{survey}-bright-{target}.flagged.dat'),
           np.array([flagged_hpix, flagged_targid]).T, fmt='%i %i')

