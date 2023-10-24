### Importing useful packages
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import glob
import cv2
import os
import argparse
from scipy.stats import norm
import datetime
from mpl_point_clicker import clicker

### Importing useful scripts
from llsfit import *

### Defining a Gaussian distribution of total counts
### such that the integral over all space is 'A'
def gaussian(x, A, sigma, mu):
	return (A*np.exp(-(1/2)*(((x - mu) / sigma)**2))) / (sigma*np.sqrt(2*np.pi))

### Defining a Gaussian distribution for two peaks
def double_gaussian(x, A1, sigma1, mu1, A2, sigma2, mu2):
	return gaussian(x, A1, sigma1, mu1) + gaussian(x, A2, sigma2, mu2)

### Get Gaussian fit to data and integrate
def gaussian_fit(x, y, sigFigs=4):
	
    ## Find peak value of line
    max_count = np.amax(y)

	## Find half the peak
    half_max = max_count//2
	
    ## Find all points above half the peak
    half_max_mask = (y >= half_max)

	## Find start and end X
    startX = x[half_max_mask][0]
    endX = x[half_max_mask][-1]
	
    ## Estimate standard deviation with FWHM
    std_estimate = (endX - startX)/2.2

	## Find energy of spectral line
    peak_energy = x[np.where(y == max_count)[0][0]]

	## Estimate for integral for area A
    a_estimate = 2.2 * max_count * std_estimate

    print('Estimated fit parameters:')
    print('A = {:.7g}'.format(a_estimate))
    print('σ = {:.7g}'.format(std_estimate))
    print('μ = {:.7g}\n'.format(peak_energy))

    ## Fit the gaussian to the ROI
    params_gauss, params_covariance = curve_fit(gaussian, xs, ys, p0=[a_estimate, std_estimate, peak_energy])

    ## Make sure fit parameters are positive
    params_gauss = np.absolute(params_gauss)

    print('Computed fit parameters:')
    print('A = {:.7g}'.format(params_gauss[0]))
    print('σ = {:.7g}'.format(params_gauss[1]))
    print('μ = {:.7g}\n'.format(params_gauss[2]))
    
    return params_gauss, params_covariance

### Integrate under the Gaussian to get total counts
def integrate_gaussian(x, y, params_gauss, sigmas=3):
	
    # Find starting x-value (μ - sigmas*σ)
    startX = params_gauss[2] - sigmas*params_gauss[1]
	
    # Find ending x-value (μ + sigmas*σ)
    endX = params_gauss[2] - sigmas*params_gauss[1]
	
    # Get mask corresponding to this range of x-values
    mask = (x >= startX) & (x <= endX)
	
    # Slice y-values within this range
    y_slice = y[mask]
	
    # Sum counts
    total_counts = np.sum(y_slice)

    ### Return total counts
    return total_counts

    