import os, sys 
import numpy as np 


def pSMF(targ, zmin, zmax): 
    ''' deploy pSMF fitting script 
    '''
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J psmf.%s.z%.1f_%.1f' % (targ, zmin, zmax), 
        "#SBATCH --mem=12G", 
        '#SBATCH --time=02:00:00',
        "#SBATCH --export=ALL",
        '#SBATCH -o o/psmf.%s.z%.1f_%.1f' % (targ, zmin, zmax), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "#SBATCH --gres=gpu:1" 
        '',
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        '', 
        'python /home/chhahn/projects/provabgs/svda/bin/psmf.py %s %.1f %.1f' % (targ, zmin, zmax), 
        ''])
        
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(a)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


pSMF('bgs_bright', 0.0, 0.1)
pSMF('bgs_bright', 0.1, 0.2)
pSMF('bgs_bright', 0.2, 0.3)
pSMF('bgs_bright', 0.3, 0.4)
