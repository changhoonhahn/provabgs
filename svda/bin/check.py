import os, sys 
import datetime
import numpy as np

import svda as SVDA


target  = 'BGS_BRIGHT'
survey  = 'sv3' 

# read healpixs and targetids
hpixels, targetids = np.loadtxt(f'/global/cfs/cdirs/desi/users/chahah/provabgs/svda/{survey}-bright-{target}.flagged.dat', 
                                dtype=int, unpack=True, usecols=[0,1])
hpixels     = hpixels.astype(int)
targetids   = targetids.astype(int)

ngals = len(hpixels)
 
n = 0 
for igal in range(ngals): 
    hpix        = hpixels[igal]
    targetid    = targetids[igal] 

    fmcmc = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/healpix/',
            str(hpix), 'provabgs.%i.hdf5' % targetid) 

    if os.path.isfile(fmcmc) and datetime.datetime.fromtimestamp(os.path.getmtime(fmcmc)).month >= 7:
        if datetime.datetime.fromtimestamp(os.path.getmtime(fmcmc)).day >= 25: 
            pass #print('already re-run %s' % fmcmc)
    else: 
        print('%i not re-run %s' % (igal, fmcmc)) 
        n += 1 
print(n)
