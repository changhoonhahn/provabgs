import os, sys 
import numpy as np 


def deploy_batch(ibsatch n_cpu=32, niter=3000, max_hr=48): 
    ''' deploy provabgs on a batch of LOWZ targets
    '''
    hr = time_healpix(hpix, target, survey, n_cpu)
    hr = np.min([hr, max_hr]) 

    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH --qos=regular',
        '#SBATCH --time=%s:00:00' % str(hr).zfill(2),
        '#SBATCH --constraint=cpu',
        '#SBATCH -N 1',
        '#SBATCH -J %i_%s_%s' % (hpix, survey, target),
        '#SBATCH -o o/%i_%s_%s.o' % (hpix, survey, target),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', 
        'source ~/.bashrc', 
        'conda activate gqp', 
        'module unload PrgEnv-intel', 
        'module load PrgEnv-gnu', 
        "",
        'export OMP_NUM_THREADS=1', 
        '', 
        'python -W ignore /global/homes/c/chahah/projects/provabgs/bin/lowz/sed.py' % (ibatch, niter, n_cpu), 
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


deploy_batch(0, n_cpu=32, niter=3000, max_hr=6) 
