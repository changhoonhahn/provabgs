import os, sys 
import glob 
import numpy as np 
from astropy import table as aTable

import svda as SVDA 

dir_fuji = '/global/cfs/cdirs/desi/spectro/redux/fuji/'


def time_petal(tileid, petal, target, survey, n_cpu): 
    ''' calculate how long the run will have to be based on number of targets
    in tile and petal 
    '''
    # read BGS targets from specified petal  
    meta, _, _, _, _, _, _, _, _ = SVDA.cumulative_tile_petal(
            tileid, ipetal, target=target, redux='fuji', survey=survey)
    ngal = len(meta)
    return int(np.ceil(0.33 * np.ceil(float(ngal) / float(n_cpu))) )


def check_file(tileid, i_petal): 
    ''' check that coadd file for tileid and i_petal exists 
    '''
    subdirs = glob.glob(os.path.join(dir_fuji, 'tiles', 'cumulative', str(tileid), '*'))
    assert len(subdirs) == 1

    fcoadd = glob.glob(os.path.join(subdirs[0], 'coadd-%i-%i-*.fits' % (i_petal, tileid)))[0]
    return os.path.isfile(fcoadd) 


def deploy_petal(tileid, petal, target, survey, n_cpu=32, niter=3000): 
    '''
    '''
    hr = time_petal(tileid, petal, target, survey, n_cpu)
    assert hr < 24

    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH -p regular',
        '#SBATCH -N 1',
        '#SBATCH -t %s:00:00' % str(hr).zfill(2),
        '#SBATCH -C haswell',
        '#SBATCH -J %i_%i_%s_%s' % (tileid, petal, survey, target),
        '#SBATCH -o o/%i_%i_%s_%s.o' % (tileid, petal, survey, target),
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
        'python -W ignore /global/homes/c/chahah/projects/provabgs/bin/svda/sed_petal.py %i %i %s %s %i %i' % (tileid, petal, target, survey, niter, n_cpu), 
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


tiles_fuji = aTable.Table.read(os.path.join(dir_fuji, 'tiles-fuji.fits'))
is_bright = (tiles_fuji['PROGRAM'] == 'bright')
is_sv3 = (tiles_fuji['SURVEY'] == 'sv3')
tileid = tiles_fuji['TILEID'][is_bright & is_sv3][np.argmax(tiles_fuji[is_bright & is_sv3]['EFFTIME_SPEC'])]

for ipetal in range(1): 
    if check_file(tileid, ipetal): 
         deploy_petal(tileid, ipetal, 'BGS_BRIGHT', 'sv3', n_cpu=32, niter=3000)
