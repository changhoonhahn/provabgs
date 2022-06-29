'''

script for deploying jobs on perlmutter 

'''
import os
import glob
import numpy as np 


def mstar_zmax(hpix, sample='sv3-bright', target='BGS_BRIGHT'): 
    '''
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH --qos=regular", 
        "#SBATCH --time=12:00:00", 
        "#SBATCH --constraint=cpu", 
        "#SBATCH -N 1", 
        "#SBATCH -J mstar_zmax_%i" % hpix,  
        "#SBATCH -o o/mstar_zmaxs_%i.o" % hpix,
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load python",
        "conda activate gqp", 
        "",
        "python /global/homes/c/chahah/projects/provabgs/svda/bin/compile_mstar_zmax.py %s %i %s" % (sample, hpix, target), 
        "", 
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_job.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _job.slurm')
    os.system('rm _job.slurm')
    return None 

# compile Mstar posteriors and zmax values 
hpixs = [int(fpost.split('-')[-1].split('.')[0]) for fpost in glob.glob('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/provabgs-*BGS_BRIGHT.hdf5')]
for hpix in hpixs[5:20]: 
    mstar_zmax(hpix)
