'''

compress training set into PCA components and save to pickle file 

'''
import os, sys 
import pickle
import numpy as np
# --- speculator ---
from speculator import SpectrumPCA


dat_dir='/global/cscratch1/sd/chahah/provabgs/' # hardcoded to NERSC directory 


def divide_trainingset_3wavebins(name, batch0, batch1): 
    ''' divide DESI training set into 3 wavelength intervals: 
      1. wave < 4500
      2. 4500 < wave < 6500
      3. 6500 < wave 
    '''
    # fsps wavelength 
    fwave   = os.path.join(dat_dir, 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 4500) 
    wave_bin1 = (wave >= 4500) & (wave < 6500) 
    wave_bin2 = (wave >= 6500) 

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    fspecs  = [os.path.join(dat_dir, 
        'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) for ibatch in batches]

    for fspec in fspecs: 
        spec = np.load(fspec)
        np.save(fspec.replace('.npy', '.w0.npy'), spec[:,wave_bin0])
        np.save(fspec.replace('.npy', '.w1.npy'), spec[:,wave_bin1])
        np.save(fspec.replace('.npy', '.w2.npy'), spec[:,wave_bin2])

    return None 


def train_pca_3wavebins(name, batch0, batch1, n_pca, i_bin): 
    ''' train PCA for DESI simpledust or complexdust training sets for 3 wavelength
    intervals: 
      1. wave < 4500
      2. 4500 < wave < 6500
      3. 6500 < wave 

    '''
    # fsps wavelength 
    fwave   = os.path.join(dat_dir, 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 4500) 
    wave_bin1 = (wave >= 4500) & (wave < 6500) 
    wave_bin2 = (wave >= 6500) 

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    
    # parameters 
    fthetas = [os.path.join(dat_dir, 'fsps.%s.theta.seed%i.npy' % (name,
        ibatch)) for ibatch in batches]
    
    wave_bin = [wave_bin0, wave_bin1, wave_bin2][i_bin]

    # log(spectra) over wavelength bin 
    fspecs  = [os.path.join(dat_dir, 
        'fsps.%s.lnspectrum.seed%i.w%i.npy' % (name, ibatch, i_bin)) for
        ibatch in batches]

    if name == 'nmf_bases': 
        # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
        n_param = 10 
    elif model == 'nmfburst': 
        # theta = [b1, b2, b3, b4, fburst, tburst, g1, g2, dust1, dust2, dust_index, zred]
        n_param = 12 
    n_wave = np.sum(wave_bin) 
    
    # train PCA basis 
    PCABasis = SpectrumPCA(
            n_parameters=n_param,       # number of parameters
            n_wavelengths=n_wave,       # number of wavelength values
            n_pcas=n_pca,               # number of pca coefficients to include in the basis 
            spectrum_filenames=fspecs,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=fthetas, # list of filenames containing the corresponding parameter values
            parameter_selection=None) 

    PCABasis.compute_spectrum_parameters_shift_and_scale() # computes shifts and scales for (log) spectra and parameters
    PCABasis.train_pca()
    PCABasis.transform_and_stack_training_data(
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.w%i.pca%i' % (name, batch0, batch1, i_bin, n_pca)), 
            retain=True) 
    # save to file 
    PCABasis._save_to_file(
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.w%i.pca%i.hdf5' % (name, batch0, batch1, i_bin, n_pca))
            )
    return None 


if __name__=="__main__": 
    job     = sys.argv[1]
    name    = sys.argv[2]
    ibatch0 = int(sys.argv[3])
    ibatch1 = int(sys.argv[4])
    
    if job == 'train': 
        n_pca = int(sys.argv[5]) 
        i_bin = int(sys.argv[6])
        train_pca_3wavebins(model, ibatch0, ibatch1, n_pca, i_bin)
    elif job == 'divide': 
        divide_trainingset_3wavebins(name, ibatch0, ibatch1) 
