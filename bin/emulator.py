import os, sys
import pickle
import numpy as np
import tensorflow as tf

from speculator import SpectrumPCA
from speculator import Speculator

#-------------------------------------------------------
# params 
#-------------------------------------------------------
model = sys.argv[1]
i_wave = int(sys.argv[2]) 
n_pcas = int(sys.argv[3]) 
Nlayer = int(sys.argv[4]) 
Nunits = int(sys.argv[5]) 
#-------------------------------------------------------
dat_dir='/scratch/gpfs/chhahn/provabgs/' # hardcoded to tiger directory 
wave = np.load(os.path.join(dat_dir, 'wave_fsps.npy')) 

wave_bins = [(wave < 4500), ((wave >= 4500) & (wave < 6500)), (wave >= 6500)]

n_hidden = [Nunits for i in range(Nlayer)]
#-------------------------------------------------------
if model == 'nmf_bases': n_param = 10
elif model == 'nmfburst': n_param = 12

# load trained PCA basis object
print('training PCA bases')
PCABasis = SpectrumPCA(
        n_parameters=n_param,       # number of parameters
        n_wavelengths=np.sum(wave_bins[i_wave]),       # number of wavelength values
        n_pcas=n_pcas,              # number of pca coefficients to include in the basis
        spectrum_filenames=None,  # list of filenames containing the (un-normalized) log spectra for training the PCA
        parameter_filenames=[], # list of filenames containing the corresponding parameter values
        parameter_selection=None) # pass an optional function that takes in parameter vector(s) and returns True/False for any extra parameter cuts we want to impose on the training sample (eg we may want to restrict the parameter ranges)
PCABasis._load_from_file(
        os.path.join(dat_dir, 
            'fsps.%s.seed0_499.w%i.pca%i.hdf5' % (model, i_wave, n_pcas)))

#-------------------------------------------------------
# training theta and pca 
_training_theta = np.load(os.path.join(dat_dir,
    'fsps.%s.seed0_499.w%i.pca%i_parameters.npy' % (model, i_wave, n_pcas)))
_training_pca = np.load(os.path.join(dat_dir,
    'fsps.%s.seed0_499.w%i.pca%i_pca.npy' % (model, i_wave, n_pcas)))

training_theta = tf.convert_to_tensor(_training_theta.astype(np.float32))
training_pca = tf.convert_to_tensor(_training_pca.astype(np.float32))


# train Speculator
speculator = Speculator(
        n_parameters=n_param, # number of model parameters
        wavelengths=wave[wave_bins[i_wave]], # array of wavelengths
        pca_transform_matrix=PCABasis.pca_transform_matrix,
        parameters_shift=PCABasis.parameters_shift,
        parameters_scale=PCABasis.parameters_scale,
        pca_shift=PCABasis.pca_shift,
        pca_scale=PCABasis.pca_scale,
        spectrum_shift=PCABasis.spectrum_shift,
        spectrum_scale=PCABasis.spectrum_scale,
        n_hidden=n_hidden, # network architecture (list of hidden units per layer)
        restore=False,
        optimizer=tf.keras.optimizers.Adam()) # optimizer for model training

#-------------------------------------------------------
# train speculator

# cooling schedule
lr = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
batch_size = [1000, 5000, 10000, 50000, 1000000, int(training_theta.shape[0])]
gradient_accumulation_steps = [1, 1, 1, 1, 10, 10] # split the largest batch size into 10 when computing gradients to avoid memory overflow

# early stopping set up
patience = 40

# writeout loss 
_floss = os.path.join(dat_dir, 
        'fsps.%s.seed0_499.w%i.pca%i.%ix%i.loss.dat' % (model, i_wave, n_pcas, Nlayer, Nunits))
floss = open(_floss, 'w')
floss.close()


# train using cooling/heating schedule for lr/batch-size
for i in range(len(lr)):
    print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]))
    # set learning rate
    speculator.optimizer.lr = lr[i]

    n_training = training_theta.shape[0]
    # create iterable dataset (given batch size)
    training_data = tf.data.Dataset.from_tensor_slices((training_theta, training_pca)).shuffle(n_training).batch(batch_size[i])

    # set up training loss
    validation_loss = [np.infty]
    best_loss       = np.infty
    early_stopping_counter = 0
    
    # loop over epochs
    while early_stopping_counter < patience:

        # loop over batches
        for theta, pca in training_data:

            # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
            if gradient_accumulation_steps[i] == 1:
                loss = speculator.training_step(theta, pca)
            else:
                loss = speculator.training_step_with_accumulated_gradients(theta, pca, accumulation_steps=gradient_accumulation_steps[i])

        # compute validation loss at the end of the epoch
        _loss = speculator.compute_loss(training_theta, training_pca).numpy()
        validation_loss.append(_loss)
    
        floss = open(_floss, "a") # append
        floss.write('%i \t %f \t %f \n' % (batch_size[i], lr[i], _loss))
        floss.close()

        # early stopping condition
        if validation_loss[-1] < best_loss:
            best_loss = validation_loss[-1]
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            speculator.update_emulator_parameters()
            speculator.save(os.path.join(dat_dir,
                '_fsps.%s.seed0_499.w%i.pca%i.%ix%i.log' % 
                (model, i_wave, n_pcas, Nlayer, Nunits)))

            attributes = list([
                    list(speculator.W_),
                    list(speculator.b_),
                    list(speculator.alphas_),
                    list(speculator.betas_),
                    speculator.pca_transform_matrix_,
                    speculator.pca_shift_,
                    speculator.pca_scale_,
                    speculator.spectrum_shift_,
                    speculator.spectrum_scale_,
                    speculator.parameters_shift_,
                    speculator.parameters_scale_,
                    speculator.wavelengths])

            # save attributes to file
            f = open(os.path.join(dat_dir, 
                'fsps.%s.seed0_499.w%i.pca%i.%ix%i.log.pkl' % 
                (model, i_wave, n_pcas, Nlayer, Nunits)), 'wb')
            pickle.dump(attributes, f)
            f.close()
            print('Validation loss = %s' % str(best_loss))

