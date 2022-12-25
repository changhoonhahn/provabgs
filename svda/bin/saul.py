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
    dat_dir = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/healpix/',
            str(hpix))
    if not os.path.isdir(dat_dir): os.system('mkdir -p %s' % dat_dir)

    fpetal = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/', 
            'provabgs-%s-bright-%i.%s.hdf5' % (survey, hpix, target)) 

    if os.path.isfile(fpetal): 
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
        '', #'module load python', 
        'conda activate gqp', 
        'module unload PrgEnv-intel', 
        'module load PrgEnv-gnu', 
        "",
        'export OMP_NUM_THREADS=1', 
        '', 
        'python -W ignore /global/homes/c/chahah/projects/provabgs/svda/bin/provabgs_healpix.py %i %s %s %i %i' % (hpix, target, survey, niter, n_cpu), 
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


def deploy_redo(target, survey, i0, i1, n_cpu=32, niter=3000, hr=48): 
    ''' deploy provabgs on flagged galaxies and targets 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        '#SBATCH --qos=regular',
        '#SBATCH --time=%s:00:00' % str(hr).zfill(2),
        '#SBATCH --constraint=cpu',
        '#SBATCH -N 1',
        '#SBATCH -J redo_%s_%s_%i_%i' % (survey, target, i0, i1),
        '#SBATCH -o o/redo_%s_%s_%i_%i.o' % (survey, target, i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        '', #'module load python', 
        'conda activate gqp', 
        'module unload PrgEnv-intel', 
        'module load PrgEnv-gnu', 
        "",
        'export OMP_NUM_THREADS=1', 
        '', 
        'python -W ignore /global/homes/c/chahah/projects/provabgs/svda/bin/provabgs_redo.py %s %s %i %i %i %i' % (target, survey, niter, n_cpu, i0, i1), 
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
    dat_dir = '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/healpix/'

    meta, _, _, _, _, _, _, _, _ = \
            SVDA.healpix(hpix, target=target, redux='fuji', survey=survey)
    ngals = len(meta)
    
    print('%i: %i' % (hpix, ngals))
    if ngals == 0: return None  

    for igal in range(ngals): 
        fmcmc = os.path.join(dat_dir, str(hpix), 
                'provabgs.%i.hdf5' % meta['TARGETID'][igal])
        if not os.path.isfile(fmcmc) and not os.path.islink(fmcmc): 
            print('  HEALPIX %i %i of %i done' % (hpix, igal, ngals))
            return True 

        post_i = Infer.PostOut()
        try: 
            post_i.read(fmcmc)
        except: 
            print('  HEALPIX %i rerunning %i of %i' % (hpix, igal, ngals))
            return True

        if igal == 0: 
            targetid    = []
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

        targetid.append(meta['TARGETID'][igal]) 
        samples.append(post_i.samples)
        log_prob.append(post_i.log_prob)
        redshift.append(post_i.redshift)

        wavelength_obs.append(post_i.wavelength_obs)
        flux_spec_obs.append(post_i.flux_spec_obs)
        ivar_spec_obs.append(post_i.ivar_spec_obs)
        flux_photo_obs.append(post_i.flux_photo_obs)
        ivar_photo_obs.append(post_i.ivar_photo_obs)

        flux_spec_model.append(post_i.flux_spec_model)
        flux_photo_model.append(post_i.flux_photo_model)
    
    fpetal = os.path.join('/global/cfs/cdirs/desi/users/chahah/provabgs/svda/', 
            'provabgs-%s-bright-%i.%s.hdf5' % (survey, hpix, target)) 
    petal = h5py.File(fpetal, 'w')
    petal.create_dataset('targetid', data=np.array(targetid))
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
    return int(np.ceil(1.5 * np.ceil(float(ngal) / float(n_cpu))) )


################################################################
# BGS BRIGHT
################################################################
#tiles_fuji = aTable.Table.read('/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/tilepix.fits') 
#is_bright = (tiles_fuji['PROGRAM'] == 'bright')
#is_sv3 = (tiles_fuji['SURVEY'] == 'sv3')
#
#hpixs = np.unique(np.sort(np.array(tiles_fuji['HEALPIX'][is_bright & is_sv3])))
#
#for hpix in hpixs: 
#    gather_healpix(hpix, target='BGS_BRIGHT', survey='sv3', n_cpu=32,
#            niter=3000, max_hr=12)

#for hpix in [9791, 9932, 9985, 10145, 10383, 10520, 11349, 15359, 16042, 25599,
#        25919, 25926, 25935, 25939, 25961, 25964, 25982, 26001, 26004, 26005,
#        26005, 26006, 26007, 26048, 26087, 26088, 26089, 26092, 26094, 27262]: 
#    meta, _, _, _, _, _, _, _, _ = \
#            SVDA.healpix(hpix, target='BGS_BRIGHT', redux='fuji', survey='sv3')
#    ngals = len(meta)
#    print('%i: %i gals' % (hpix, ngals))

# re-run select BGS Bright galaxies 
for i in range(11):  #3333
    deploy_redo('BGS_BRIGHT', 'sv3', i*300, (i+1)*300, n_cpu=32, niter=3000, hr=12) 


################################################################
# BGS FAINT 
################################################################
#tiles_fuji = aTable.Table.read('/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/tilepix.fits') 
#is_bright = (tiles_fuji['PROGRAM'] == 'bright')
#is_sv3 = (tiles_fuji['SURVEY'] == 'sv3')
#
#hpixs = np.unique(np.sort(np.array(tiles_fuji['HEALPIX'][is_bright & is_sv3])))
#print('%i healpix total' % len(hpixs))

#for hpix in hpixs: 
#    print('>>> %s' % hpix)
#    gather_healpix(hpix, target='BGS_FAINT', survey='sv3', n_cpu=32,
#            niter=3000, max_hr=12)

# re-run select BGS Faint galaxies 
for i in range(13): # 3855
    deploy_redo('BGS_FAINT', 'sv3', i*300, (i+1)*300, n_cpu=32, niter=3000, hr=12) 
