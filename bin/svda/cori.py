import os, sys 
import h5py
import glob 
import numpy as np 
from astropy import table as aTable

import svda as SVDA 

dir_fuji = '/global/cfs/cdirs/desi/spectro/redux/fuji/'


#####################################################################################
# healpix
#####################################################################################
def gather_healpix(hpix, target='BGS_BRIGHT', survey='sv3', n_cpu=32,
        niter=3000, max_hr=48): 
    ''' gather all the posteriors for BGS targets in healpix
    '''
    dat_dir = os.path.join('/global/cscratch1/sd/chahah/provabgs/svda/healpix/',
            str(hpix))
    if not os.path.isdir(dat_dir): os.system('mkdir -p %s' % dat_dir)

    fpetal = os.path.join(dat_dir, str(hpix), 
            'provabgs-%s-bright-%i.hdf5' % (survey, hpix)) 
    if os.path.isfile(fpetal): 
        print('%s done' % fpetal)
        return None 

    re_run = _gather_healpix_posteriors(hpix, target, survey, niter=niter)

    if re_run: 
        deploy_healpix(hpix, target, survey, n_cpu=n_cpu, niter=niter,
                max_hr=max_hr)
    return None 


def deploy_healpix(hpix, target, survey, n_cpu=32, niter=3000, max_hr=48): 
    '''
    '''
    hr = time_healpix(hpix, target, survey, n_cpu)
    hr = np.min([hr, max_hr]) 

    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH -p regular',
        '#SBATCH -N 1',
        '#SBATCH -t %s:00:00' % str(hr).zfill(2),
        '#SBATCH -C haswell',
        '#SBATCH -J %i_%s_%s' % (hpix, survey, target),
        '#SBATCH -o o/%i_%s_%s.o' % (hpix, survey, target),
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
        'python -W ignore /global/homes/c/chahah/projects/provabgs/bin/svda/sed_healpix.py %i %s %s %i %i' % (hpix, target, survey, niter, n_cpu), 
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


def _gather_healpix_posteriors(hpix, target, survey, niter=3000): 
    ''' compile all the posteriors for a petal.
    '''
    from provabgs import infer as Infer
    dat_dir = '/global/cscratch1/sd/chahah/provabgs/svda/healpix/'

    meta, _, _, _, _, _, _, _, _ = \
            SVDA.healpix(hpix, target=target, redux='fuji', survey=survey)
    ngals = len(meta)
    
    if ngals == 0: return None  

    for igal in range(ngals): 
        fmcmc = os.path.join(dat_dir, str(hpix), 
                'provabgs.%i.hdf5' % meta['TARGETID'][igal])
        if not os.path.isfile(fmcmc): 
            print('  %i of %i done' % (igal, ngals))
            return True 

        post_i = Infer.PostOut()
        post_i.read(fmcmc)

        if igal == 0: 
            samples     = [] 
            log_prob    = [] 
            redshift    = []

            wavelength_obs  = [] 
            flux_spec_obs   = [] 
            ivar_spec_obs   = []
            flux_photo_obs  = []
            ivar_photo_obs  = []

            flux_spec_model     = []
            flux_photo_model    = [] 
        else: 
            samples.append(post_i.samples)
            log_prob.append(post_i.log_prob)
            redshift.append(post_i.redshift)

            wavelength_obs.append(post_i.wavelength_obs)
            flux_spec_obs.append(post_i.flux_spec_obs)
            ivar_spec_obs.append(post_i.ivar_spec_obs)
            flux_photo_obs.append(post_i.flux_photo_obs)
            ivar_photo_obs.append(post_i.ivar_photo_obs)

            flux_spec_model.append(post_i.flux_spec_model)
            flux_photo_model.append(post_i.flux_spec_model)
    
    fpetal = os.path.join(dat_dir, str(hpix), 
            'provabgs-%s-bright-%i.%s.hdf5' % (survey, hpix, target)) 
    petal = h5py.File(fpetal, 'w')
    petal.create_dataset('samples', data=np.array(samples))
    petal.create_dataset('log_prob', data=np.array(log_prob))
    petal.create_dataset('redshift', data=np.array(redshift))

    petal.create_dataset('wavelength_obs', data=np.array(wavelength_obs))
    petal.create_dataset('flux_spec_obs', data=np.array(flux_spec_obs))
    petal.create_dataset('ivar_spec_obs', data=np.array(ivar_spec_obs))
    petal.create_dataset('flux_photo_obs', data=np.array(flux_photo_obs))
    petal.create_dataset('ivar_photo_obs', data=np.array(ivar_photo_obs))

    petal.create_dataset('flux_spec_model', data=np.array(flux_spec_model))
    petal.create_dataset('flux_photo_model', data=np.array(flux_photo_model))
    petal.close() 

    # delete individual posterior files
    '''
        for igal in range(ngals): 
            fmcmc = os.path.join(dat_dir, str(tileid), 
                    'provabgs.%i.hdf5' % meta['TARGETID'][igal])
            os.system('rm %s' % fmcmc)
    '''
    return False


def time_healpix(hpix, target, survey, n_cpu): 
    ''' calculate how long the run will have to be based on number of targets
    in tile and petal 
    '''
    # read BGS targets from specified petal  
    meta, _, _, _, _, _, _, _, _ = \
            SVDA.healpix(hpix, target=target, redux='fuji', survey=survey)
    ngal = len(meta)
    print(ngal)
    return int(np.ceil(1.5 * np.ceil(float(ngal) / float(n_cpu))) )


#####################################################################################
def deploy_petals(tileid, target='BGS_BRIGHT', survey='sv3', n_cpu=32, niter=3000): 
    ''' deploy provabgs on specified targets in the 10 petals of tile 
    ''' 
    for ipetal in range(10): 
        if check_file(tileid, ipetal): 
             deploy_petal(tileid, ipetal, target, survey, n_cpu=n_cpu, niter=n_iter)
    return None 


def gather_petals(tileid, target='BGS_BRIGHT', survey='sv3', n_cpu=32, niter=3000): 
    ''' gather all the provabgs petal files 
    '''
    dat_dir = os.path.join('/global/cscratch1/sd/chahah/provabgs/svda/',
            str(tileid))
    if not os.path.isdir(dat_dir): os.system('mkdir -p %s' % dat_dir)
    # check that all provabgs posterior file exists for target 
    for ipetal in range(10): 
        fpetal = os.path.join( dat_dir, 'provabgs.%i.%i.hdf5' % (tileid, ipetal)) 
        if os.path.isfile(fpetal): continue

        re_run = _gather_posteriors(tileid, ipetal, target, survey, niter=niter)

        if re_run: 
            deploy_petal(tileid, ipetal, target, survey, n_cpu=n_cpu, niter=niter)
    return None 


def _gather_posteriors(tileid, ipetal, target, survey, niter=3000): 
    ''' compile all the posteriors for a petal.
    '''
    from provabgs import infer as Infer
    dat_dir = '/global/cscratch1/sd/chahah/provabgs/svda/'

    meta, _, _, _, _, _, _, _, _ = SVDA.cumulative_tile_petal(tileid, ipetal, 
            target=target, redux='fuji', survey=survey)
    ngals = len(meta)
    
    for igal in range(ngals): 
        fmcmc = os.path.join(dat_dir, str(tileid), 
                'provabgs.%i.hdf5' % meta['TARGETID'][igal])
        if not os.path.isfile(fmcmc): 
            return True 

        post_i = Infer.PostOut()
        post_i.read(fmcmc)

        if igal == 0: 
            samples     = [] 
            log_prob    = [] 
            redshift    = []

            wavelength_obs  = [] 
            flux_spec_obs   = [] 
            ivar_spec_obs   = []
            flux_photo_obs  = []
            ivar_photo_obs  = []

            flux_spec_model     = []
            flux_photo_model    = [] 
        else: 
            samples.append(post_i.samples)
            log_prob.append(post_i.log_prob)
            redshift.append(post_i.redshift)

            wavelength_obs.append(post_i.wavelength_obs)
            flux_spec_obs.append(post_i.flux_spec_obs)
            ivar_spec_obs.append(post_i.ivar_spec_obs)
            flux_photo_obs.append(post_i.flux_photo_obs)
            ivar_photo_obs.append(post_i.ivar_photo_obs)

            flux_spec_model.append(post_i.flux_spec_model)
            flux_photo_model.append(post_i.flux_spec_model)
    
    fpetal = os.path.join( dat_dir, str(tileid), 
            'provabgs.%i.%i.hdf5' % (tileid, ipetal)) 
    petal = h5py.File(fpetal, 'w')
    petal.create_dataset('samples', data=np.array(samples))
    petal.create_dataset('log_prob', data=np.array(log_prob))
    petal.create_dataset('redshift', data=np.array(redshift))

    petal.create_dataset('wavelength_obs', data=np.array(wavelength_obs))
    petal.create_dataset('flux_spec_obs', data=np.array(flux_spec_obs))
    petal.create_dataset('ivar_spec_obs', data=np.array(ivar_spec_obs))
    petal.create_dataset('flux_photo_obs', data=np.array(flux_photo_obs))
    petal.create_dataset('ivar_photo_obs', data=np.array(ivar_photo_obs))

    petal.create_dataset('flux_spec_model', data=np.array(flux_spec_model))
    petal.create_dataset('flux_photo_model', data=np.array(flux_photo_model))
    petal.close() 

    # delete individual posterior files
    '''
        for igal in range(ngals): 
            fmcmc = os.path.join(dat_dir, str(tileid), 
                    'provabgs.%i.hdf5' % meta['TARGETID'][igal])
            os.system('rm %s' % fmcmc)
    '''
    return False


def time_petal(tileid ,ipetal, target, survey, n_cpu): 
    ''' calculate how long the run will have to be based on number of targets
    in tile and petal 
    '''
    # read BGS targets from specified petal  
    meta, _, _, _, _, _, _, _, _ = SVDA.cumulative_tile_petal(
            tileid, ipetal, target=target, redux='fuji', survey=survey)
    ngal = len(meta)
    return int(np.ceil(1.5 * np.ceil(float(ngal) / float(n_cpu))) )


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


#tiles_fuji = aTable.Table.read(os.path.join(dir_fuji, 'tiles-fuji.fits'))
#is_bright = (tiles_fuji['PROGRAM'] == 'bright')
#is_sv3 = (tiles_fuji['SURVEY'] == 'sv3')
#
#deepest_tiles = tiles_fuji['TILEID'][is_bright & is_sv3][np.argsort(tiles_fuji[is_bright & is_sv3]['EFFTIME_SPEC'])[::-1]]
#
#for tileid in deepest_tiles[:10]: 
#    print(tileid) 
#    gather_petals(tileid, target='BGS_BRIGHT', survey='sv3', n_cpu=32, niter=3000)

tiles_fuji = aTable.Table.read('/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/tilepix.fits') 
is_bright = (tiles_fuji['PROGRAM'] == 'bright')
is_sv3 = (tiles_fuji['SURVEY'] == 'sv3')

hpixs = np.unique(np.sort(np.array(tiles_fuji['HEALPIX'][is_bright & is_sv3])))

for hpix in hpixs[:100]: 
    gather_healpix(hpix, target='BGS_BRIGHT', survey='sv3', n_cpu=32,
            niter=3000, max_hr=6)
