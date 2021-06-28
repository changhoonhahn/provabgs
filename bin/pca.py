'''

compress training set into PCA components and save to pickle file 

'''
import os, sys 
import pickle
import numpy as np
# --- speculator ---
from speculator import SpectrumPCA

if os.environ['machine'] == 'cori': 
    dat_dir='/global/cscratch1/sd/chahah/provabgs/emulator/' # hardcoded to NERSC directory 
elif os.environ['machine'] == 'tiger': 
    dat_dir='/tigress/chhahn/provabgs/'

version = '0.1'

name    = sys.argv[1]
batch0  = int(sys.argv[2])
batch1  = int(sys.argv[3])
n_pca   = int(sys.argv[4]) 
i_bin   = int(sys.argv[5])

# fsps wavelength 
fwave   = os.path.join(dat_dir, 'wave.%s.npy' % name) 
wave    = np.load(fwave)

# wavelength bins  
wave_bin = [ 
        (wave >= 1000) & (wave < 2000), 
        (wave >= 2000) & (wave < 3600), 
        (wave >= 3600) & (wave < 5500), 
        (wave >= 5500) & (wave < 7410), 
        (wave >= 7410) & (wave < 60000)
        ][i_bin]

str_wbin = [
        '.w1000_2000', 
        '.w2000_3600', 
        '.w3600_5500', 
        '.w5500_7410', 
        '.w7410_60000' 
        ][i_bin]

# batches of fsps spectra
batches = range(batch0, batch1+1)

# parameters 
if name == 'nmf': 
    fthetas = [os.path.join(dat_dir, 'fsps.%s.v%s.theta_unt.seed%i.npy' % (name, version, ibatch)) for ibatch in batches]
else:  
    fthetas = [os.path.join(dat_dir, 'fsps.%s.v%s.theta.seed%i.npy' % (name, version, ibatch)) for ibatch in batches]

# log(spectra) over wavelength bin 
fspecs  = [
        os.path.join(dat_dir, 'fsps.%s.v%s.lnspectrum.seed%i%s.npy' % (name, version, ibatch, str_wbin)) 
        for ibatch in batches]

if name == 'nmf': # theta = [b1, b2, b3, b4, g1, g2, dust1, dust2, dust_index, zred]
    n_param = 10
elif name == 'burst': # theta = [tburst, zburst, dust1, dust2, dust_index]
    n_param = 4

n_wave  = np.sum(wave_bin) 
fpca    = os.path.join(dat_dir, 'fsps.%s.v%s.seed%i_%i%s.pca%i.hdf5' % (name, version, batch0, batch1, str_wbin, n_pca))
print(fpca)

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
PCABasis.transform_and_stack_training_data(fpca.replace('.hdf5', ''), retain=True) 
# save to file 
PCABasis._save_to_file(fpca)
