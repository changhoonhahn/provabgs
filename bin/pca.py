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


def divide_training(name, batch0, batch1): 
    ''' divide training set into 4 wavelenght intervals; 
    1. wave  < 3600 (UV-NUV wavelenght bins dlambda = 20A) 
    2. 3600 < wave < 5500 (optical wavelength bins dlambda = 0.9)
    3. 5500 < wave < 7410 (optical wavelength bins dlambda = 0.9)
    3. 7410 < wave < 59900 (NIR-IR wavelength bins dlambda > 20)
    '''
    dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 

    # fsps wavelength 
    fwave   = os.path.join(dat_dir, 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 3600) 
    wave_bin1 = (wave >= 3600) & (wave < 5500) 
    wave_bin2 = (wave >= 5500) & (wave < 7410) 
    wave_bin3 = (wave >= 7410) 

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    fspecs  = [os.path.join(dat_dir, 
        'fsps.%s.lnspectrum.seed%i.npy' % (name, ibatch)) for ibatch in batches]

    for fspec in fspecs: 
        spec = np.load(fspec)
        np.save(fspec.replace('.npy', '.w0.npy'), spec[:,wave_bin0])
        np.save(fspec.replace('.npy', '.w1.npy'), spec[:,wave_bin1])
        np.save(fspec.replace('.npy', '.w2.npy'), spec[:,wave_bin2])
        np.save(fspec.replace('.npy', '.w3.npy'), spec[:,wave_bin3])
    return None 


def train_pca_wavebin(name, batch0, batch1, n_pca, i_bin): 
    ''' train PCA for provabgs training set for specified wavelength interal. 
    '''
    if os.environ['machine'] == 'cori': 
        dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 
    elif os.environ['machine'] == 'tiger': 
        dat_dir='/tigress/chhahn/provabgs/'
    # fsps wavelength 
    fwave   = os.path.join(dat_dir, 'wave_fsps.npy') 
    wave    = np.load(fwave)

    # wavelength bins  
    wave_bin0 = (wave < 3600) 
    wave_bin1 = (wave >= 3600) & (wave < 5500) 
    wave_bin2 = (wave >= 5500) & (wave < 7410) 
    wave_bin3 = (wave >= 7410) 

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    
    # parameters 
    fthetas = [os.path.join(dat_dir, 'fsps.%s.theta.seed%i.npy' % (name,
        ibatch)) for ibatch in batches]
    
    wave_bin = [wave_bin0, wave_bin1, wave_bin2, wave_bin3][i_bin]

    # log(spectra) over wavelength bin 
    fspecs  = [os.path.join(dat_dir, 
        'fsps.%s.lnspectrum.seed%i.w%i.npy' % (name, ibatch, i_bin)) for
        ibatch in batches]

    if name == 'nmf': # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
        n_param = 10
    elif name == 'burst': # theta = [tburst, zburst, dust1, dust2, dust_index]
        n_param = 5

    n_wave = np.sum(wave_bin) 

    print(os.path.join(dat_dir, 'fsps.%s.seed%i_%i.3w%i.pca%i.hdf5' % (name, batch0, batch1, i_bin, n_pca)))
    
    # train PCA basis 
    PCABasis = SpectrumPCA(
            n_parameters=n_param,       # number of parameters
            n_wavelengths=n_wave,       # number of wavelength values
            n_pcas=n_pca,               # number of pca coefficients to include in the basis 
            spectrum_filenames=fspecs,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=fthetas, # list of filenames containing the corresponding parameter values
            parameter_selection=None) 
    print('compute spectrum parameters shift and scale') 
    PCABasis.compute_spectrum_parameters_shift_and_scale() # computes shifts and scales for (log) spectra and parameters
    print('train pca') 
    PCABasis.train_pca()
    print('transform and stack') 
    PCABasis.transform_and_stack_training_data(
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.3w%i.pca%i' % (name, batch0, batch1, i_bin, n_pca)), 
            retain=True) 
    # save to file 
    PCABasis._save_to_file(
            os.path.join(dat_dir, 'fsps.%s.seed%i_%i.3w%i.pca%i.hdf5' % (name, batch0, batch1, i_bin, n_pca))
            )
    return None 


if __name__=="__main__": 
    job     = sys.argv[1]
    name    = sys.argv[2]
    ibatch0 = int(sys.argv[3])
    ibatch1 = int(sys.argv[4])
    
    if job == 'divide': 
        divide_training(name, ibatch0, ibatch1) 
    elif job == 'train': 
        n_pca = int(sys.argv[5]) 
        i_bin = int(sys.argv[6])

        train_pca_wavebin(name, ibatch0, ibatch1, n_pca, i_bin)
