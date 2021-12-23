import os, sys 
import numpy as np 

def deploy_sed(igal0, igal1, sample='sv_everest_faint_sf.fits', n_cpu=32, niter=3000): 
    '''
    '''
    samp = sample.split('.')[0].split('_')[-1]
    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH -p regular',
        '#SBATCH -N 1',
        '#SBATCH -t 06:00:00',
        '#SBATCH -C haswell',
        '#SBATCH -J %s_%i_%i' % (samp, igal0, igal1),
        '#SBATCH -o o/_%s_%i_%i.o' % (samp, igal0, igal1),
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
        'python -W ignore /global/homes/c/chahah/projects/provabgs/bin/raga/sed.py %i %i %s %i %i' % (igal0, igal1, sample, niter, n_cpu), 
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

n_sf    = 48731
n_agn   = 2087
n_comp  = 2135

# 12/23/2021 --- first test run
#deploy_sed(0, 31, sample='sv_everest_faint_sf.fits', n_cpu=32, niter=3000)
for i in range(1,(n_agn // 32)+1): 
    deploy_sed(32*i, np.min([32*(i+1)-1, n_agn-1]), sample='sv_everest_faint_agn.fits', n_cpu=32, niter=3000)
for i in range(1,(n_comp // 32)+1): 
    deploy_sed(32*i, np.min([32*(i+1)-1, n_agn-1]), sample='sv_everest_faint_comp.fits', n_cpu=32, niter=3000)
