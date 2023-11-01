#import os
#import re
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.optimize import curve_fit
#import glob
#import cv2
#import argparse
#from scipy.stats import norm
#import datetime
from mpl_point_clicker import clicker

import utils.gaussian_fit as gauss


def plot_data(x, y, plotArgs, xBounds=None, yBounds=None):
	"""
	Plots the given data with the specified arguments.

	Args:
		x (list or array): The x-axis data.
		y (list or array): The y-axis data.
		plotArgs (dict): A dictionary of arguments to pass to the plot function.
		xBounds (tuple, optional): The x-axis bounds to plot. Defaults to None.
		yBounds (tuple, optional): The y-axis bounds to plot. Defaults to None.

	Returns:
		None
	"""
	# Clear existing plots
	plt.clf()

	# Plot whole spectrum
	plt.plot(x, y, color=plotArgs['color'], label=plotArgs['label'])

	# If user specified x-bounds
	if xBounds != None:

		# Set x-bounds
		plt.xlim(xBounds)

	# If user specified y-bounds
	if yBounds != None:

	# Set y-bounds
		plt.ylim(yBounds)

		# Add title
		plt.title(plotArgs['title'])

		# Add axes labels
		plt.xlabel(plotArgs['xlabel'])
		plt.ylabel(plotArgs['ylabel'])

	# If user wants to show a legend
	if plotArgs['legend']:

		# Add legend
		plt.legend()
	
def select_peak(bins, spectral_data, element, clipVal=0, doublet=False):
	"""
    Selects a peak from a given spectrum and fits a Gaussian curve to it.

    Args:
        bins (numpy.ndarray): Array of bin values for the spectrum.
        spectrum (numpy.ndarray): Array of counts for each bin in the spectrum.
        element (str): Name of the element being analyzed.
        clipVal (float, optional): Value to clip the spectrum at. Defaults to 0.
        doublet (bool, optional): Whether the peak is part of a doublet. Defaults to False.

    Returns:
        tuple: Tuple containing the fit parameters for the Gaussian curve, the covariance matrix, and the integrated counts at the peak.
    """

	## If user wants to clip the data
	if clipVal != 0:

		# Clip to remove noise (optional)
		calibrated_spectrum = np.clip(spectral_data, a_min = 0, a_max = clipVal)

	## If no clip value is given
	else:

		# Check how many bins the spectrum contains
		nBins = len(spectral_data)

		# Clipping noise (about bin 100 or 1/10 of the spectrum)
		cutoff = int(nBins/10)

		# Find the largest value in the spectrum after cuffoff
		clipVal = np.max(spectral_data[cutoff:])*1.5

		# Clip spectrum at this point and remove noise
		spectral_data = np.clip(spectral_data, a_min = 0, a_max = clipVal)

	## Plot and display the spectrum
	plotArgs = {
		'color': 'k',
		'label': 'data',
		'xlabel': 'bin',
		'ylabel': 'counts',
		'title': f'{element}: Zoom into peak',
		'legend': True
	}

	## Zoom into the region of interest
	# Plot spectrum
	plot_data(bins, spectral_data, plotArgs)

	# Get input points from user (click 2 points on the display)
	points = np.asarray(plt.ginput(2))

	# Close plot
	plt.close()

	# Get start and end 'x' coordinates
	startX = points[0][0]
	endX = points[1][0]

	# Define x-bounds
	xBounds = (startX, endX)

	# For the y-bounds, mask spectrum and find maximum value
	mask = (bins > startX)*(bins < endX)
	masked_spectrum = spectral_data[mask]
	peak_height = np.max(masked_spectrum)

	# Define y-bounds
	startY = (-1)*peak_height*0.2
	endY = peak_height*1.2
	yBounds = (startY, endY)

	# Plot zoomed in area
	plotArgs['title'] = f'{element}: Fit a peak'
	plot_data(bins, spectral_data, plotArgs, xBounds=xBounds, yBounds=yBounds)

	## Zooming further into the region for higher precision
	# Get input points from user (click 2 points on the display)
	points = np.asarray(plt.ginput(2))

	# Close plot
	plt.close()

	# Get start and end 'x' coordinates 
	startX = points[0][0]
	endX = points[1][0]

	## Obtain fit parameters for the Gaussian fit
	width = endX - startX
	extraR = 0.2

	print(f'Fitting region from bin {startX:.2f} to find {endX:.2f}.\n')

	## Plotting final fit over region
	plotArgs['title'] = f'{element}: Gaussian fit to spectral line.'
	plot_data(bins, spectral_data, plotArgs, xBounds=xBounds, yBounds=yBounds)

	# Masking the bins and calibrated spectrum
	mask = (bins > startX)*(bins < endX)
	masked_bins = bins[mask]
	masked_spectrum = spectral_data[mask]

	# Fit the Gaussian to the region of interest (see gaussian_fit.py)
	params_gauss, params_covariance, max_counts = gauss.gaussian_fit(masked_bins, masked_spectrum)

	int_counts = gauss.integrate_gaussian(bins, spectral_data, params_gauss, sigmas=3)

	print(f'Integrated counts at peak: {int_counts:.4g}')

	## Get counts for the fit curve
	fit_counts = gauss.gaussian(masked_bins, *params_gauss)

	# Plot gaussian fit of region of interest
	plt.plot(masked_bins, fit_counts, color='red', label='Gaussian fit')
	plt.show()

	## If doublet, the start and end x-values are needed
	if doublet:
		return params_gauss, params_covariance, int_counts, (startX, endX), max_counts

	### Return fit parameters
	return params_gauss, params_covariance, int_counts, max_counts	

def display_spectrum():
	#spectral_data, calibration_path, save=False
	"""
	Calibrates and displays the given spectral data.

	Args:
		spectral_data (list or array): A list or array of spectral data.
		calibration_path (str): The path to the calibration curve file.
		save (bool, optional): Whether to save the plot. Defaults to False.

	Returns:
		None
	"""
	print("Displaying spectrum...")

	# ... do something with the spectral data ...
	return


def nist_data(coords, units='keV'):
	"""
	Retrieves NIST X-ray data for a given set of coordinates.

	Args:
		coords (list): A list of coordinates to retrieve data for.
		units (str, optional): The units to return the data in. Defaults to 'keV'.

	Returns:
		dict: A dictionary containing the retrieved data.
	"""
	## Print header
	print("--------------------------------------")
	print(f"index | energy ({units:3}) | relative counts")
	print("--------------------------------------")

    ## Go through every coordinate
	for i, coord in enumerate(coords):

        # Print coordinate with index
		print(f"{i+1:5} | {coords[i][0]:12} | {coords[i][1]}")
    
	print("--------------------------------------")

def isotope_styles(x, y, input_isotope):
	# Clear existing plots
	plt.clf()

	isotopes = ['Am', 'Ba', 'Fe', 'Zn', 'Cd']

	if input_isotope in isotopes:
		isotope_styles = {
			'Am': {'marker': 'o', 'color': 'yellow'},
			'Ba': {'marker': 's', 'color': 'tab:purple'},
			'Fe': {'marker': 'v', 'color': 'r'},
			'Zn': {'marker': '^', 'color': 'b'},
			'Cd': {'marker': '*', 'color': 'g'}
		}
	
		marker = isotope_styles[input_isotope]['marker']
		color = isotope_styles[input_isotope]['color']


	else:
		# Define markers and colors
		markers = ['o', 's', '^', 'v', '*']
		colors = ['yellow', 'tab:purple', 'r', 'b', 'g']

		# Create a dictionary mapping isotopes to markers and colors
		isotope_styles = {input_isotope: {'marker': marker, 'color': color} 
					for input_isotope, marker, color in zip(input_isotope, markers, colors)}
	
	plt.scatter(x, y, label=input_isotope, zorder=2, marker=marker, color=color, edgecolors='k')
