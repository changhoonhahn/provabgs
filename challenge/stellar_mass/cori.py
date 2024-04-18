import os, sys 
import numpy as np 
from astropy.table import Table 


def deploy_bgs_test(i0, i1, n_cpu=32, niter=3000, hr=48): 
    ''' deploy provabgs on flagged galaxies and targets 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH -p regular',
        '#SBATCH -N 1',
        '#SBATCH -t %s:00:00' % str(hr).zfill(2),
        '#SBATCH -C haswell',
        '#SBATCH -J bgs_test_%i_%i' % (i0, i1),
        '#SBATCH -o o/_best_test_%i_%i.o' % (i0, i1),
        '#SBATCH -L SCRATCH,project',
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', 
        'conda activate gqp', 
        'module unload PrgEnv-intel', 
        'module load PrgEnv-gnu', 
        "",
        'export OMP_NUM_THREADS=1', 
        '', 
        'python -W ignore /global/homes/c/chahah/projects/provabgs/challenge/stellar_mass/run_sed.py %i %i %i %i' % (i0, i1, niter, n_cpu), 
        '', 
        'now=$(date +"%T")',
        'echo "end time ... $now"', 
        ""]) 
        
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(cntnt)
    f.close()

    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


for i in range(1, 99): 
    deploy_bgs_test(510*i, np.min([510*(i+1), 50578]), n_cpu=32, niter=3000, hr=24)
