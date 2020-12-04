'''

some utility functions 

'''
import os
import numba
import numpy as np
from astropy.io import fits 


# --- constants ---- 
def Lsun(): 
    return 3.846e33  # erg/s


def parsec(): 
    return 3.085677581467192e18  # in cm


def to_cgs(): # at 10pc 
    lsun = Lsun()
    pc = parsec()
    return lsun/(4.0 * np.pi * (10 * pc)**2) 


def c_light(): # AA/s
    return 2.998e18


def jansky_cgs(): 
    return 1e-23


def desi_Resolution(res_data): 
    from desispec.resolution import Resolution
    return Resolution(res_data)

# --- the code below is taken from the `redrock` python package. I've copied
# the code over instead of importing it to reduce package dependencies.
# https://github.com/desihub/redrock/blob/8e33f642b6d8a762c4d71626fb3aca6377c976d4/py/redrock/rebin.py#L10
# --- 
def centers2edges(centers):
    """Convert bin centers to bin edges, guessing at what you probably meant
    Args:
        centers (array): bin centers,
    Returns:
        array: bin edges, lenth = len(centers) + 1
    """
    centers = np.asarray(centers)
    edges = np.zeros(len(centers)+1)
    #- Interior edges are just points half way between bin centers
    edges[1:-1] = (centers[0:-1] + centers[1:]) / 2.0
    #- edge edges are extrapolation of interior bin sizes
    edges[0] = centers[0] - (centers[1]-edges[1])
    edges[-1] = centers[-1] + (centers[-1]-edges[-2])

    return edges

# This code is purposely written in a very "C-like" way.  The logic
# being that it may help numba optimization and also makes it easier
# if it ever needs to be ported to Cython.  Actually Cython versions
# of this code have already been tested and shown to perform no better
# than numba on Intel haswell and KNL architectures.

@numba.jit
def _trapz_rebin(x, y, edges, results):
    '''
    Numba-friendly version of trapezoidal rebinning
    See redrock.rebin.trapz_rebin() for input descriptions.
    `results` is pre-allocated array of length len(edges)-1 to keep results
    '''
    nbin = len(edges) - 1
    i = 0  #- index counter for output
    j = 0  #- index counter for inputs
    yedge = 0.0
    area = 0.0

    while i < nbin:
        #- Seek next sample beyond bin edge
        while x[j] <= edges[i]:
            j += 1

        #- What is the y value where the interpolation crossed the edge?
        yedge = y[j-1] + (edges[i]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])

        #- Is this sample inside this bin?
        if x[j] < edges[i+1]:
            area = 0.5 * (y[j] + yedge) * (x[j] - edges[i])
            results[i] += area

            #- Continue with interior bins
            while x[j+1] < edges[i+1]:
                j += 1
                area = 0.5 * (y[j] + y[j-1]) * (x[j] - x[j-1])
                results[i] += area

            #- Next sample will be outside this bin; handle upper edge
            yedge = y[j] + (edges[i+1]-x[j]) * (y[j+1]-y[j]) / (x[j+1]-x[j])
            area = 0.5 * (yedge + y[j]) * (edges[i+1] - x[j])
            results[i] += area

        #- Otherwise the samples span over this bin
        else:
            ylo = y[j] + (edges[i]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            yhi = y[j] + (edges[i+1]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            area = 0.5 * (ylo+yhi) * (edges[i+1]-edges[i])
            results[i] += area

        i += 1

    for i in range(nbin):
        results[i] /= edges[i+1] - edges[i]

    return

def trapz_rebin(x, y, xnew=None, edges=None):
    """Rebin y(x) flux density using trapezoidal integration between bin edges
    Notes:
        y is interpreted as a density, as is the output, e.g.
        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, edges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])
    Args:
        x (array): input x values.
        y (array): input y values.
        edges (array): (optional) new bin edges.
    Returns:
        array: integrated results with len(results) = len(edges)-1
    Raises:
        ValueError: if edges are outside the range of x or if len(x) != len(y)
    """
    if edges is None:
        edges = centers2edges(xnew)
    else:
        edges = np.asarray(edges)

    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be within input x range')

    result = np.zeros(len(edges)-1, dtype=np.float64)

    _trapz_rebin(x, y, edges, result)

    return result
