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
from gauss import *
from llsfit import *

### Utilities for repo:

### Get numerical input
def getNumericalInput(text):
	## Define temporary variable
	num = 'temp'
	## Check if number inputted
	while not num.isnumeric():
		# Get input
		num = input(text)
		# If this is not a numerical value
		if not num.isnumeric():
			print('Input a number!')
	## Return input as float
	return float(num)

### Read spectrum file and return calibration data and spectrum
def readSpectrumFile(filePath):

	### Open the spectrum file
	with open(filePath) as file:

		## Read file
		lines = file.read().splitlines()

		## Lists to store lines from file
		calibrationData = []
		spectralData = []

		## Flag to check whether current line is part of spectrum
		spectrum = False

		## Iterate through all lines
		for line in lines:

			# Check if we are currently in <<DATA> section
			if line == '<<DATA>>':

				spectrum = True

				continue

			# Check if we are at the end of spectrum
			if line == '<<END>>':

				spectrum = False

				continue

			# If neither, add to required list
			else:

				# If line part of calibration data
				if not spectrum:

					calibrationData.append(line)

				# If line part of spectrum
				if spectrum:

					spectralData.append(line)

	### Return calibration data and spectrum
	return calibrationData, np.asarray(spectralData).astype(int)

### Calibrate spectrum with 
def calibrateSpectrum(calibrationData, spectralData):

	## Flag to keep track of where in the calibration data we are
	flag = 0

	## List to store calibration 'coordinates'
	calibrationCoords = []

	## Go through every line in calibration data
	for line in calibrationData:

		# If we are in the calibration section
		if line == "<<CALIBRATION>>":

			# Set correct flag
			flag = 1

		# If we are in ROI section
		if line == "<<ROI>>":

			# Set in correct flag
			flag = 0

		# If we are in ROI section
		if line == "<<DP5 CONFIGURATION>>":

			# Set in correct flag
			flag = 0

		# If flag is greater than 1
		if flag >= 1:

			# If flag is 2 we are looking at energy units
			if flag == 2:

				# Get units (either keV or eV)
				energyUnit = line.split(' ')[-1]

			# If flag is 3, we are looking at calibration 'coordinates'
			if flag >= 3:

				# Obtain current 'coordinate'
				coordinate = np.asarray(line.split(' ')).astype(float)

				# Add current 'coordinate' to list
				calibrationCoords.append(coordinate)

			# Keep incrementing flag to keep track of the line
			flag = flag + 1

			continue

	## Convert calibration coordinates to NumPy array
	calibrationCoords = np.asarray(calibrationCoords)

	## If calibration coords were found
	if len(calibrationCoords) != 0:

		## Finding slope from coordinates. Note that the first two coordinates are used
		m = (calibrationCoords[1][1] - calibrationCoords[0][1])/(calibrationCoords[1][0] - calibrationCoords[0][0])

	## If calibration coords were not found
	else:

		##
		m = 1

	if 'energyUnit' not in locals():
		energyUnit = 'keV'

	## Check units for energy. Rescale if in eV
	if energyUnit == 'eV':
		m = m/1000

	## Get required number of bins
	binCount = len(spectralData)

	## Create calibrated bins
	bins = np.linspace(0*m, (binCount-1)*m, binCount)

	# bins = ((25-22.1)/(5.537-4.916))*(bins-4.916) + 22.1

	## Convert spectrum to integer type
	calibratedSpectrum = spectralData.astype(int)

	## Output bins and spectrum
	return bins, calibratedSpectrum

## Function to easily plot data
def plotData(xs, ys, plotArgs, xBounds=None, yBounds=None):

	# Clear existing plots
	plt.clf()

	# Plot whole spectrum
	plt.plot(xs, ys, color=plotArgs['color'], label=plotArgs['label'])

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

### Fit peak in spectrum manually
def obtainPeak(binning, spectrum, element, clipVal=0):

	## First, we will deal with the noise. There are two ways to do this.
	## Either the user has already defined a clip value, and all values above
	## will be clipped to remove thermal noise.

	## If user wants to clip the data
	if clipVal != 0:

		# Clip the spectrum to remove noise (optional)
		calibratedSpectrum = np.clip(spectrum, a_min = 0, a_max = clipVal)

	## Or no clip value has been specified. In this case, thhe code will
	## Extrapolate a clip value using the data.

	## If no clip value has been provided
	else:

		# Check how many bins the spectrum contains
		nBins = len(spectrum)

		# A good guess for where the noise drops to zero is around bin 100.
		# This is about a tenth of the way into the spectrum.

		# Extrapolate a cutoff
		cutoff = int(nBins/10)

		# Find the largest value in the spectrum beyond the cutoff.
		# Set the clip value to 1.5 times this.
		clipVal = np.max(spectrum[cutoff:])*1.5

		# Clip the spectrum to remove noise
		spectrum = np.clip(spectrum, a_min = 0, a_max = clipVal)

	## We will start by plotting the uncalibrated spectrum for the user to
	## choose the region to zoom into. This will be done by plotting the spectrum,
	## Getting two input points which delimit the x-values, and y-values for the
	## plot will be extrapolated from the curve inside the x-values.

	## Create dictionary to store plotting parameters
	plotArgs = {
		'color': 'k',
		'label': 'data',
		'xlabel': 'bin',
		'ylabel': 'counts',
		'title': f'{element}: Zoom into peak',
		'legend': True
	}

	## Plot data
	plotData(binning, spectrum, plotArgs)

	## Get point inputs
	points = np.asarray(plt.ginput(2))

	## Close the plot
	plt.close()

	## Get start and end 'x' coordinates
	startX = points[0][0]
	endX = points[1][0]

	## Define x-bounds
	xBounds = (startX, endX)

	## Mask to find y-bounds
	mask = (binning > startX)*(binning < endX)

	## Masked spectrum
	maskedSpectrum = spectrum[mask]

	## Maximum value in masked spectrum
	peakHeight = np.max(maskedSpectrum)

	## y-bounds for plot
	startY = (-1)*peakHeight*0.2
	endY = peakHeight*1.2
	yBounds = (startY, endY)

	## Now that we have obtained the region to zoom into, the spectrum will be
	## plotted again but this time x-values will be obtained using input points
	## to delimit the region to fit a Gaussian.

	## Change title of plot
	plotArgs['title'] = f'{element}: Fit a peak'

	## Plot data
	plotData(binning, spectrum, plotArgs, xBounds=xBounds, yBounds=yBounds)

	## Get point inputs
	points = np.asarray(plt.ginput(2))

	## Close the plot
	plt.close()

	## Get start and end 'x' coordinates
	startX = points[0][0]
	endX = points[1][0]

	## Now that we have these x-values delimiting the Gaussian, the curve can be fit
	## and we can obtain the fitting parameters.

	## Width from start to end
	width = endX - startX

	## Extra region to plot (in percent)
	extraR = 0.2

	print(f'Fitting region from bin {startX:.2f} to bin {endX:.2f}.\n')

	## Title text
	plotArgs['title'] = f'{element}: Gaussian fit to spectral line'

	## Plot data
	plotData(binning, spectrum, plotArgs, xBounds=xBounds, yBounds=yBounds)

	## Create mask for certain values
	mask = (binning > startX)*(binning < endX)

	## Mask the bins and calibrated spectrum
	maskedBins = binning[mask]
	maskedSpectrum = spectrum[mask]

	## Fit the gaussian to the ROI
	popt, pcov = getGaussFit(maskedBins, maskedSpectrum)

	## Get integral of Gaussian
	intCounts = intGauss(binning, spectrum, popt, sigmas=3)

	print('Integrated counts at peak: {:.4g}'.format(intCounts))

	## Get counts for the fit curve
	fitCounts = gaussian(maskedBins, *popt)

	## Plot gaussian fit of ROI
	plt.plot(maskedBins, fitCounts, color='red', label='Gaussian fit')

	## Show plots
	plt.show()

	## Return fit parameters
	return popt

### Prints out a spectrum to select peaks
def printSpectrum(coords, units='keV'):

	## Print header
	print("--------------------------------------")
	print(f"index | energy ({units:3}) | relative counts")
	print("--------------------------------------")

	## Go through every coordinate
	for i, coord in enumerate(coords):

		# Print coordinate with index
		print(f"{i+1:5} | {coords[i][0]:12} | {coords[i][1]}")

	print("--------------------------------------")
