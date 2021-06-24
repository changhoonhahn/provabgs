'''
'''
import os
import numpy as np 
from provabgs import models as Models


def test_DESIspeculator():
    ''' script to test the trained speculator model for DESI 
    '''
    # initiate desi model 
    Mdesi = Models.DESIspeculator() 
    
    # load test parameter and spectrum 
    test_theta = np.load('/Users/chahah/data/gqp_mc/speculator/DESI_complexdust.theta_test.npy') 
    test_logspec = np.load('/Users/chahah/data/gqp_mc/speculator/DESI_complexdust.logspectrum_fsps_test.npy') 
    
    for i in range(10): 
        print(1.-(Mdesi._emulator(test_theta[i]) - np.exp(test_logspec[i]))/np.exp(test_logspec[i]))
        print('')
    return None 



if __name__=="__main__": 
    test_DESIspeculator() 
