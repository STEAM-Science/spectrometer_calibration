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
from utils import *

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


### Defining a Gaussian distribution of total counts
### such that the integral over all space is 'A'
def gaussian(xs, A, sigma, mu):

	temp = (xs - mu) / sigma
	temp = -(1/2)*(temp**2)
	temp = np.exp(temp)
	return (A*np.exp(-(1/2)*(((xs - mu) / sigma)**2))) / (sigma*np.sqrt(2*np.pi))#(A*temp) / (sigma*np.sqrt(2*np.pi))


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


## Function to easily plot data
def plotData(xs, ys, plotArgs):

	# Clear existing plots
	plt.clf()

	# Plot whole spectrum
	plt.plot(xs, ys, color=plotArgs['color'], label=plotArgs['label'])

	# Add title
	plt.title(plotArgs['title'])

	# plt.xlim(19,30)

	# Add axes labels
	plt.xlabel(plotArgs['xlabel'])
	plt.ylabel(plotArgs['ylabel'])

	# If user wants to show a legend
	if plotArgs['legend']:

		# Add legend
		plt.legend()


### Fit peak in spectrum manually
def obtainPeak(binning, spectrumUncalibrated, clipVal=0):

	## If user wants to clip the data
	if clipVal != 0:

		# Clip the spectrum to remove noise (optional)
		calibratedSpectrum = np.clip(spectrumUncalibrated, a_min = 0, a_max = clipVal)

	## Create dictionary to store plotting parameters
	plotArgs = {
		'color': 'k',
		'label': 'data',
		'xlabel': 'bin',
		'ylabel': 'counts',
		'title': 'uncalibrated spectrum',
		'legend': True
	}

	## Plot data
	plotData(binning, spectrumUncalibrated, plotArgs)

	## Get point inputs
	points = np.asarray(plt.ginput(2))

	## Get start and end 'x' coordinates
	startX = points[0][0]
	endX = points[1][0]

	## Width from start to end
	width = endX - startX

	## Extra region to plot (in percent)
	extraR = 0.2

	print(f'Fitting region from bin {startX} to bin {endX}.\n')

	## Close the plot
	plt.close()

	## Title text
	plotArgs['title'] = 'spectral line'

	## Plot data
	plotData(binning, spectrumUncalibrated, plotArgs)

	## Plot the correct ranges
	plt.xlim(startX - width*(extraR/2), endX + width*(extraR/2))

	## Create mask for certain values
	mask = (binning > startX)*(binning < endX)

	## Mask the bins and calibrated spectrum
	maskedBins = binning[mask]
	maskedSpectrum = spectrumUncalibrated[mask]

	## Fit the gaussian to the ROI
	popt, pcov = getGaussFit(maskedBins, maskedSpectrum)

	## Get integral of Gaussian
	intCounts = intGauss(binning, spectrumUncalibrated, popt, sigmas=3)

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


### Create calibration for an element using known spectrum
def createCalibration(filePath):

	## Find element from filePath
	element = filePath.split('/')[-1].split('_')[-2]

	## If directory for calibration curve outputs does not exist
	if not os.path.isdir("./cCurves/"):

		# Make directory for calibration files
		os.mkdir("./cCurves/")

	## Check how many calibration curves for element already exist
	cCurves = glob.glob("cCurves/*.txt")

	## Only choose file paths that contain element
	cCurves = [cPath for csvPath in cCurves if element in cPath]

	## Number of current curve
	cNum = len(cCurves) + 1

	print(f'Creating calibration file {cNum} for {element}.')

	## Load selected spectrum from filePath
	meta, spectrumUncalibrated = readSpectrumFile(filePath)

	## Obtain binning
	binning = np.arange(len(spectrumUncalibrated))

	## Filepath to corresponding NIST spectrum
	nistPath = f"NISTData/Spectral_Lines_{element}.csv"

	## Read NIST data
	xsNIST, ysNIST = np.loadtxt(nistPath, delimiter=',').T

	## Create mask to clip data to below 50 keV
	mask = xsNIST < 50

	## Clip data to below 50 keV
	xsNIST = xsNIST[mask]
	ysNIST = ysNIST[mask]

	## Recompose into matrix of coords
	NISTCoords = np.array([xsNIST, ysNIST]).T

	### Show the spectrum (uncalibrated)

	## Get an input of number of peaks
	numPeaks = 0
	while numPeaks < 2:
		numPeaks = int(getNumericalInput('Input number of peaks to fit (minimum 2 required): '))

	## Create list to store calibration points
	calibration = []

	## For every peak in the range
	for nPeak in range(1, numPeaks+1):

		print(f'\n\nChoose peak {nPeak} from {element} spectrum.')

		## Display NIST coords
		printSpectrum(NISTCoords, units='keV')

		print('\nDisplaying uncalibrated spectrum. Close once a peak has been chosen.')

		## Create dictionary to store plotting parameters
		plotArgs = {
			'color': 'k',
			'label': 'data',
			'xlabel': 'bin',
			'ylabel': 'counts',
			'title': 'uncalibrated spectrum',
			'legend': True
		}
		
		## Plotting uncalibrated spectrum
		plotData(binning, spectrumUncalibrated, plotArgs)

		## Showing plot
		plt.show()

		## Get an input of peak index
		pNum = int(getNumericalInput('Input index of NIST peak to fit: '))

		## Obtain fit gaussian from spectrum
		A, sigma, mean = obtainPeak(binning, spectrumUncalibrated)

		## Get energy of fit peak
		pEnergy = NISTCoords[pNum-1, 0]

		## Add pair of calibration point to array
		calibration.append(np.array([pNum, pEnergy]))

	## Convert calibration points to array
	calibration = np.asarray(calibration)

	## Get list of 'x' and 'y' calibration points
	calibration = calibration.T

	## Get calibration curve
	m = m_unweighted(calibration[0], calibration[1])
	b = b_unweighted(calibration[0], calibration[1])
	mErr = sig_m_unweighted(calibration[0], calibration[1])
	bErr = sig_b_unweighted(calibration[0], calibration[1])

	print(f'\nFinal calibration curve: y = {m:.3g}x + {b:.3g}')
			
	## Get current date and time
	dt = datetime.datetime.now()
	date = dt.strftime("%Y_%m_%d")

	## Create dictionary of calibration values
	calibration = {
		'm': np.array([m]),
		'b': np.array([b]),
		'mErr': np.array([mErr]),
		'bErr': np.array([bErr]),
	}

	df = pd.DataFrame.from_dict(calibration)

	## Create output address for calibration file
	outPath = f"cCurves/{element}_{date}_curve_{str(cNum).zfill(2)}.csv"

	## Save calibration data to csv
	df.to_csv(outPath, index=False, encoding='utf-8')

	print(f'\nSaved calibration curve to {outPath}.')

	return


### Display a spectrum
def display(filePath, save=False):

	return


### Function to create csv file from spectrum data file
def createCsv(filePath):

	## Use readSpectrumFile() to read file
	calibrationData, spectralData = readSpectrumFile(filePath)

	## Calibrate output of file with calibration data
	bins, calibratedSpectrum = calibrateSpectrum(calibrationData, spectralData)

	## Create new dictionary with data
	df = {
		'energy (keV)': bins,
		'counts': calibratedSpectrum,
	}

	## Convert dictionary to pandas dataframe
	df = pd.DataFrame(df)

	## Path to output csv file
	csvPath = './csvOut/' + filePath.split('/')[-1].split('.')[0] + '.csv'

	## If directory for csv outputs does not exist
	if not os.path.isdir("./csvOut/"):

		# Make directory for csv files
		os.mkdir("./csvOut/")

	## Save csv
	df.to_csv(f"{csvPath}", index=False)

	print(f"Saved data to {csvPath}.")


### Main functioning of script
def main(args):

	## Check if user wants to create a calibration file
	if args.calibrate:

		print(f"Calibrating with {args.src}.\n")

		# Run createCalibration
		createCalibration(args.src)

		return

	## If the user wants to display spectrum
	if args.display:

		print(f"Displaying {args.src}.\n")

		# Run displaySpectrum
		displaySpectrum(args.src, save=args.save)

		return

	## If the user wants to export data to a csv
	if args.csv:

		print(f"Creating csv file from {args.src}.\n")

		# Run createCsv
		createCsv(args.src)

		return

	return


### Run if file is run directly
if __name__ == '__main__':

	## Create new parser
	parser = argparse.ArgumentParser(description='Process inputs to calibrate spectra.')

	## Choose spectrum sourse
	parser.add_argument('--src', action='store', nargs='?', type=str, default='spectra/55Fe_CdTe.txt', help='Spectrum source file.')

	## Choose whether to create spectrum calibration file
	parser.add_argument('--calibrate', action='store_true', help='Choose whether to create spectrum calibration file.')

	## Choose whether to display a spectrum
	parser.add_argument('--display', action='store_true', help='Choose whether to display a spectrum.')

	## Choose whether to save a displayed spectrum
	parser.add_argument('--save', action='store_true', help='Choose whether to save a displayed spectrum.')

	## Choose whether to create a csv from spectrum
	parser.add_argument('--csv', action='store_true', help='Choose whether to create a csv from spectrum.')

	## Parse arguments
	args = parser.parse_args()

	## Call main
	main(args)