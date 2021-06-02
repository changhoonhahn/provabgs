'''

compress training set into PCA components and save to pickle file 

'''
import os, sys 
import pickle
import numpy as np
# --- speculator ---
from speculator import SpectrumPCA

if os.environ['machine'] == 'cori': 
    dat_dir='/global/cscratch1/sd/chahah/provabgs/' # hardcoded to NERSC directory 
elif os.environ['machine'] == 'tiger': 
    dat_dir='/tigress/chhahn/provabgs/'
else: 
    raise ValueError


def divide_trainingset_3wavebins(name, batch0, batch1): 
    ''' divide DESI training set into 3 wavelength intervals: 
      1. wave < 4500
      2. 4500 < wave < 6500
      3. 6500 < wave 
    '''
    dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 

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


def divide_trainingset_6wavebins(name, batch0, batch1): 
    ''' divide DESI training set into 6 wavelength intervals
    1. (wave < 3000) 
    2. (wave >= 3000) & (wave < 4000) 
    3. (wave >= 4000) & (wave < 5000) 
    4. (wave >= 5000) & (wave < 6000) 
    5. (wave >= 6000) & (wave < 7000) 
    6. (wave >= 7000)
    '''
    # fsps wavelength 
    fwave   = os.path.join(dat_dir, 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 3000) 
    wave_bin1 = (wave >= 3000) & (wave < 4000) 
    wave_bin2 = (wave >= 4000) & (wave < 5000) 
    wave_bin3 = (wave >= 5000) & (wave < 6000) 
    wave_bin4 = (wave >= 6000) & (wave < 7000) 
    wave_bin5 = (wave >= 7000)

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    fspecs  = [os.path.join(dat_dir, 
        'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) for ibatch in batches]

    for fspec in fspecs: 
        spec = np.load(fspec)
        np.save(fspec.replace('.npy', '.6w0.npy'), spec[:,wave_bin0])
        np.save(fspec.replace('.npy', '.6w1.npy'), spec[:,wave_bin1])
        np.save(fspec.replace('.npy', '.6w2.npy'), spec[:,wave_bin2])
        np.save(fspec.replace('.npy', '.6w3.npy'), spec[:,wave_bin3])
        np.save(fspec.replace('.npy', '.6w4.npy'), spec[:,wave_bin4])
        np.save(fspec.replace('.npy', '.6w5.npy'), spec[:,wave_bin5])

    return None 


def train_pca_3wavebins(name, batch0, batch1, n_pca, i_bin): 
    ''' train PCA for DESI simpledust or complexdust training sets for 3 wavelength
    intervals: 
      1. wave < 4500
      2. 4500 < wave < 6500
      3. 6500 < wave 

    '''
    if os.environ['machine'] == 'cori': 
        dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 
    elif os.environ['machine'] == 'tiger': 
        dat_dir='/tigress/chhahn/provabgs/'
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
    elif name == 'nmfburst': 
        # theta = [b1, b2, b3, b4, fburst, tburst, g1, g2, dust1, dust2, dust_index, zred]
        n_param = 12 
    elif name == 'nmf': 
        # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
        n_param = 10
    elif name == 'burst': 
        n_param = 5

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
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.3w%i.pca%i' % (name, batch0, batch1, i_bin, n_pca)), 
            retain=True) 
    # save to file 
    PCABasis._save_to_file(
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.3w%i.pca%i.hdf5' % (name, batch0, batch1, i_bin, n_pca))
            )
    return None 


def train_pca_6wavebins(name, batch0, batch1, n_pca, i_bin): 
    ''' train PCA for DESI simpledust or complexdust training sets for 3 wavelength
    intervals: 
    '''
    # fsps wavelength 
    fwave   = os.path.join(dat_dir, 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 3000) 
    wave_bin1 = (wave >= 3000) & (wave < 4000) 
    wave_bin2 = (wave >= 4000) & (wave < 5000) 
    wave_bin3 = (wave >= 5000) & (wave < 6000) 
    wave_bin4 = (wave >= 6000) & (wave < 7000) 
    wave_bin5 = (wave >= 7000)

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    
    # parameters 
    fthetas = [os.path.join(dat_dir, 'fsps.%s.theta.seed%i.npy' % (name,
        ibatch)) for ibatch in batches]
    
    wave_bin = [wave_bin0, wave_bin1, wave_bin2, wave_bin3, wave_bin4, wave_bin5][i_bin]

    # log(spectra) over wavelength bin 
    fspecs  = [os.path.join(dat_dir, 
        'fsps.%s.lnspectrum.seed%i.6w%i.npy' % (name, ibatch, i_bin)) for
        ibatch in batches]

    if name == 'nmf_bases': 
        # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
        n_param = 10 
    elif name == 'nmfburst': 
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
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.6w%i.pca%i' % (name, batch0, batch1, i_bin, n_pca)), 
            retain=True) 
    # save to file 
    PCABasis._save_to_file(
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.6w%i.pca%i.hdf5' % (name, batch0, batch1, i_bin, n_pca))
            )
    return None 


if __name__=="__main__": 
    job     = sys.argv[1]
    name    = sys.argv[2]
    ibatch0 = int(sys.argv[3])
    ibatch1 = int(sys.argv[4])
    
    if job == 'train': 
        n_wavebins = int(sys.argv[5])
        n_pca = int(sys.argv[6]) 
        i_bin = int(sys.argv[7])

        if n_wavebins == 3: 
            train_pca_3wavebins(name, ibatch0, ibatch1, n_pca, i_bin)
        elif n_wavebins == 6: 
            train_pca_6wavebins(name, ibatch0, ibatch1, n_pca, i_bin)

    elif job == 'divide': 
        n_wavebins = int(sys.argv[5])
        if n_wavebins == 3: 
            divide_trainingset_3wavebins(name, ibatch0, ibatch1) 
        elif n_wavebins == 6: 
            divide_trainingset_6wavebins(name, ibatch0, ibatch1) 
