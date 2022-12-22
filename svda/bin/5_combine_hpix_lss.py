'''


combine healpix posterior files with LSS catalogs from kp3 


'''
import os
import h5py
import numpy as np
import astropy.table as aTable


dir_dat = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/EDAbeta/'


for target in ['BGS_BRIGHT', 'BGS_ANY']: 
    _bgs_n = aTable.Table.read(os.path.join(dir_dat, f'{target}_N_clustering.dat.fits'))
    _bgs_s = aTable.Table.read(os.path.join(dir_dat, f'{target}_S_clustering.dat.fits'))

    bgs = aTable.vstack([_bgs_n, _bgs_s])
    
    # calculate healpixel values for each LSS object 
    bgs['phi']      = np.radians(bgs['RA'])
    bgs['theta']    = np.radians(90. - bgs['DEC'])
    bgs['healpix']  = healpy.ang2pix(64, bgs['theta'], bgs_bright['phi'], nest=True)
    

    # healpixs in SV3 bright time 
    tiles_fuji = aTable.Table.read('/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/tilepix.fits')
    is_sv3 = (tiles_fuji['SURVEY'] == 'sv3')
    is_bright = (tiles_fuji['PROGRAM'] == 'bright')
    hpixs = np.unique(np.sort(np.array(tiles_fuji['HEALPIX'][is_bright & is_sv3])))
    
    # compile zmax, bestfit theta, bestfit logM*, logM* samples
    zmaxes          = np.repeat(-999., len(bgs))
    logmstars       = np.tile(-999., (len(bgs), 30))
    logmstar_bfs    = np.repeat(-999., len(bgs))
    theta_bfs       = np.tile(-999., (len(bgs), 13))

    for hpix in hpixs:
        try:
            f = h5py.File(os.path.join(
                '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/', 
                f'provabgs-{survey}-bright-{hpix}.{target}.hdf5'), 'r')

            tid     = f['targetid'][...]    # target id
            zmax    = f['zmax'][...]        # redshift max 
            ttbf    = f['theta_bf'][...]    # best-fit parameter values 
            
            # stellar mass samples  
            logms       = f['logMstar'][...].reshape((f['samples'].shape[0],
                                                      f['samples'].shape[1],
                                                      f['samples'].shape[2]))[:,:,-1]
            # stellar mass  of best-fit parmaeter 
            logms_bf    = f['logMstar_bf'][...]

        except IndexError:
            print('no %s galaxies in %i' % (target, hpix))
            continue
        except KeyError:
            print('no %s galaxies in %i' % (target, hpix))
            continue
        except OSError:
            print('no posteriors for %i' % hpix)
            continue
        except:
            print('problem reading %i' % hpix)
            continue

        is_hpix = (bgs['healpix'] == hpix)

        for i in np.arange(len(bgs))[is_hpix]:
            is_target = (tid == bgs['TARGETID'][i])
            assert np.sum(is_target) == 1

            zmaxes[i]           = zmax[is_target]
            logmstars[i,:]      = logms[is_target]
            logmstar_bf[i,:]    = logms_bf[is_target]
            theta_bfs[i,:]      = ttbf[is_target] 

    bgs['zmax']            = zmaxes
    bgs['theta_bf']        = theta_bfs
    bgs['logMstar_mcmc']   = logmstars
    bgs['logMstar_bf']     = logmstar_bf

    bgs.write(os.path.join(
        '/global/cfs/cdirs/desi/users/chahah/provabgs/svda/', 
        f'{target}_clustering.sv3.logMstar.hdf5'), overwrite=True)
