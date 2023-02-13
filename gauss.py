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
def gaussian(xs, A, sigma, mu):
	return (A*np.exp(-(1/2)*(((xs - mu) / sigma)**2))) / (sigma*np.sqrt(2*np.pi))

### Literally just two gaussians
def doubleGaussian(xs, A1, sigma1, mu1, A2, sigma2, mu2):
	return gaussian(xs, A1, sigma1, mu1) + gaussian(xs, A2, sigma2, mu2)

### Get Gaussian fit to data and integrate
def getGaussFit(xs, ys, sigFigs=4):

	## Find peak value of line
	maxCount = np.amax(ys)

	## Find half the peak
	halfMax = maxCount//2

	## Find all points above half the peak
	halfMaxMask = (ys >= halfMax)

	## Find start and end X
	startX = xs[halfMaxMask][0]
	endX = xs[halfMaxMask][-1]

	## Estimate standard deviation with FWHM
	stdEstimate = (endX - startX)/2.2

	## Find energy of spectral line
	peakEnergy = xs[np.where(ys == maxCount)[0][0]]

	## Estimate for integral
	AEstimate = 2.2*maxCount*stdEstimate

	print('Estimated fit parameters:')
	print('A = {:.7g}'.format(AEstimate))
	print('σ = {:.7g}'.format(stdEstimate))
	print('μ = {:.7g}\n'.format(peakEnergy))

	## Fit the gaussian to the ROI
	popt, pcov = curve_fit(gaussian, xs, ys, p0=[AEstimate, stdEstimate, peakEnergy])


	## Make sure fit parameters are positive
	popt = np.absolute(popt)

	print('Computed fit parameters:')
	print('A = {:.7g}'.format(popt[0]))
	print('σ = {:.7g}'.format(popt[1]))
	print('μ = {:.7g}\n'.format(popt[2]))

	## Return fit parameters
	return popt, pcov

### Integrate under the Gaussian to get total counts
def intGauss(xs, ys, popt, sigmas=3):

	## Find starting x-value (μ - sigmas*σ)
	startX = popt[2] - sigmas*popt[1]

	## Find ending x-value (μ + sigmas*σ)
	endX = popt[2] + sigmas*popt[1]

	## Get mask corresponding to this range
	mask = (xs > startX)*(xs < endX)

	## Slice y-values using the range
	ysSliced = ys[mask]

	## Add the counts up
	totCounts = np.sum(ysSliced)

	## Return total counts
	return totCounts