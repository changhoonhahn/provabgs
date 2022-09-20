import os, sys 
import numpy as np 


def deploy_batch(ibatch, n_cpu=32, niter=3000, hr=48): 
    ''' deploy provabgs on a batch of LOWZ targets
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH --qos=debug',
        '#SBATCH --time=00:15:00',
        #'#SBATCH --qos=regular',
        #'#SBATCH --time=%s:00:00' % str(hr).zfill(2),
        '#SBATCH --constraint=cpu',
        '#SBATCH -N 1',
        '#SBATCH -J lowz%i' % ibatch,
        '#SBATCH -o o/lowz%i.o' % ibatch,
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
        'python -W ignore /global/homes/c/chahah/projects/provabgs/bin/lowz/sed.py %i %i %i' % (ibatch, niter, n_cpu), 
        '', 
        'now=$(date +"%T")',
        'echo "end time ... $now"', 
        ""]) 
        
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(cntnt)
    f.close()

    os.system('sbatch script.slurm')
    #os.system('rm script.slurm')
    return None 


deploy_batch(2, n_cpu=32, niter=3000, hr=1) 
