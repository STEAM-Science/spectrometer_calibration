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
from gauss import *
from llsfit import *


### Create calibration points of known spectrum
def calibrationPoints(filePath):

	## Normalize filepath
	filePath = os.path.normpath(filePath)

	## Split filepath into individual componenets
	pathComps = filePath.split(os.sep)

	## Find element from filePath
	element = pathComps[-1].split('_')[4]

	## Filename as calibration curve
	outFile = pathComps[-1].split('.')[-2] + "_points.csv"

	## File path to create in output
	outDir = pathComps[-2]

	## Output path
	outPath = os.path.join("cPoints", outDir, outFile)

	## If directory for calibration points outputs does not exist
	if not os.path.isdir("cPoints/"):

		# Make directory for calibration files
		os.mkdir("cPoints/")

	## If directory for output directory does not exist
	if not os.path.isdir(os.path.join("cPoints", outDir)):

		# Make directory for output files
		os.mkdir(os.path.join("cPoints", outDir))

	## Check if the calibration curve already exists
	if len(glob.glob(outPath)) > 0:

		print(f'File {outFile} already exists! Delete or rename before running again.')

		print(f'\nPath to existing file: {outPath}')

		# Exit out of script
		return

	## Load selected spectrum from filePath
	meta, spectrum = readSpectrumFile(filePath)

	## Obtain binning
	binning = np.arange(len(spectrum))

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
	print('\nDisplaying uncalibrated spectrum. Close once decided how many peaks to fit.')

	## Create dictionary to store plotting parameters
	plotArgs = {
		'color': 'k',
		'label': 'data',
		'xlabel': 'bin',
		'ylabel': 'counts',
		'title': f'{element}: count peaks that will be fit and close window',
		'legend': True
	}

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
	spectrumTemp = np.clip(spectrum, a_min = 0, a_max = clipVal)
	
	## Plotting uncalibrated spectrum
	plotData(binning, spectrumTemp, plotArgs)

	## Showing plot
	plt.show()

	## Get an input of number of peaks
	numPeaks = 0
	while numPeaks < 1:
		numPeaks = int(getNumericalInput('Input number of peaks to fit (minimum 1): '))

	## Create list to store calibration points
	points = []

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
			'title': f'{element}: choose a peak and close window',
			'legend': True
		}
		
		## Plotting uncalibrated spectrum
		plotData(binning, spectrumTemp, plotArgs)

		## Showing plot
		plt.show()

		## Get an input of peak index
		pNum = int(getNumericalInput('Input index of NIST peak to fit: '))

		## Obtain fit gaussian from spectrum
		popt, intCounts = obtainPeak(binning, spectrum, element)

		## Unpacking fit parameters
		A, sigma, mean = popt

		## Get energy of fit peak
		pEnergy = NISTCoords[pNum-1, 0]

		## Add pair of calibration point to array with uncertainty
		points.append(np.array([mean, pEnergy, sigma, A, intCounts]))

	## Convert calibration points to array
	points = np.asarray(points)

	## Get list of 'x' and 'y' calibration points
	points = points.T

	## Create dictionary of calibration datapoints
	pointsDict = {
		'mu': points[0],
		'E': points[1],
		'sigma': points[2],
		'A': points[3],
		'intCounts': points[4]
	}

	## Create pandas dataframe from dictionary
	df = pd.DataFrame.from_dict(pointsDict)

	## Save calibration data to csv
	df.to_csv(outPath, index=False, encoding='utf-8')

	print(f'\nSaved calibration datapoints to {outPath}.')

### Create a calibration curve
def createCalibration():

	## If directory for calibration curve outputs does not exist
	if not os.path.isdir("cCurves/"):

		# Make directory for calibration files
		os.mkdir("cCurves/")

	## List to store files of calibration points
	pointAdds = []

	## Flag to store whether more points must be added
	morePoints = True

	## Obtain points from variety of sources
	while morePoints:

		# Get input of file address
		newFile = input('Input path to points file: ')

		# If this file does not exist
		while len(glob.glob(newFile)) == 0:

			# Get input of file address
			newFile = input('File not found! Input path to points file: ')

		# Add file address to calibration points
		pointAdds.append(newFile)

		# Ask user if more 
		morePoints = input('Add more .csv files with points? (Y/[N])')

		# If user wants to add more
		if morePoints == 'Y':
			morePoints = True
		# If user does not want to add more
		else:
			morePoints = False

	## Create arrays to store points
	mus = np.array([])
	Es = np.array([])
	sigmas = np.array([])

	## Go through every address
	for pointAdd in pointAdds:

		# Load dataframe
		df = pd.read_csv(pointAdd, index_col=False)

		# Add points to arrays
		mus = np.concatenate((mus, df['mu']))
		Es = np.concatenate((Es, df['E']))
		sigmas = np.concatenate((sigmas, df['sigma']))

	## Get calibration curve
	m = m_unweighted(mus, Es)
	b = b_unweighted(mus, Es)
	mErr = sig_m_unweighted(mus, Es)
	bErr = sig_b_unweighted(mus, Es)

	print(f'\nFinal calibration curve: y = {m:.3g}x + {b:.3g}')
	print(f'Errors:')
	print(f'm = {m:.4f} ± {mErr:.4f}')
	print(f'b = {b:.4f} ± {bErr:.4f}')

	## Calculating r-squared

	## Model-predicted energies
	modelEs = m*mus + b

	## R-squared
	rsq = 1 - (np.sum((Es - modelEs)**2)/np.sum((Es - np.mean(Es))**2))

	print(f'r-squared = {rsq:.4f}')

	## Create dictionary of calibration values
	calibration = {
		'm': np.array([m]),
		'b': np.array([b]),
		'mErr': np.array([mErr]),
		'bErr': np.array([bErr]),
		'r-squared': np.array([rsq]),
	}

	## Create pandas dataframe from dictionary
	df = pd.DataFrame.from_dict(calibration)

	## Create dictionary of calibration values
	points = {
		'mu': mus,
		'E': Es,
		'sigma': sigmas,
	}

	## Create pandas dataframe from dictionary
	dfPoints = pd.DataFrame.from_dict(points)

	## Name of file
	name = input('\nInput name of calibration curve (.csv will be added at the end): ')

	## Path to calibration curve output
	outPath = os.path.join("cCurves", name+'.csv')

	## Path to calibration curve points output
	outPathPoints = os.path.join("cCurves", name+'_points.csv')

	## Path to calibration curve graph output
	outPathFig = os.path.join("cCurves", name+'_graph.png')

	## Save calibration data to csv
	df.to_csv(outPath, index=False, encoding='utf-8')

	## Save calibration datapoints to csv
	dfPoints.to_csv(outPathPoints, index=False, encoding='utf-8')

	## Clearing plot
	plt.clf()

	## Plotting the points with fit curve
	plt.scatter(mus, Es, label='datapoints', color='k', zorder=2)

	## Points to display fit
	xsFit = np.linspace(0, 1024, num=int(1e3))
	EsFit = m*xsFit + b

	## Plotting calibration curve
	plt.plot(xsFit, EsFit, color='b', label=f'fit (y = {m:.5f} x + {b:.5f})', zorder=1)

	## Plot properties
	plt.xlabel('MCA channel')
	plt.ylabel('energy (keV)')
	plt.xlim(0, 1023)
	plt.legend()

	## Save figure
	plt.savefig(outPathFig, dpi=300)

	## Displaying plot
	plt.show()

	print(f'\nSaved calibration curve to {outPath}.')

	return

### Display a spectrum
def displaySpectrum(filePath, caliPath, save=False):

	## Load selected spectrum from filePath
	meta, spectrum = readSpectrumFile(filePath)

	## Obtain binning
	binning = np.arange(len(spectrum))

	## Load calibration curve
	caliDF = pd.read_csv(caliPath)

	## Calibration curve
	m = caliDF['m'][0]
	b = caliDF['b'][0]

	## Calibrate binning
	energies = m*binning + b

	## Split filepath into individual componenets
	pathComps = filePath.split(os.sep)

	## Find element from filePath
	element = pathComps[-1].split('_')[4]

	## Filename as plot
	outFile = pathComps[-1].split('.')[-2] + "_plot.png"

	## File path to create in output
	outDir = pathComps[-2]

	## Create dictionary to store plotting parameters
	plotArgs = {
		'color': 'b',
		'label': 'data',
		'xlabel': 'keV',
		'ylabel': 'counts',
		'title': f'{element}: spectrum',
		'legend': True
	}

	## Check if user wants to change bounds
	doBounds = input("Input x-bounds? (Y/[N]): ")

	## If user wants to input bounds
	if doBounds == 'Y':

		# Get bounds
		xStart = getNumericalInput("Input starting energy: ")
		xEnd = getNumericalInput("Input ending energy: ")

		# Set bounds
		xBounds = (xStart, xEnd)

	## Otherwise
	else:

		# Set no bounds
		xBounds = None

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

	## Plot data
	plotData(energies, spectrum, plotArgs, xBounds=xBounds)

	## Check if user wants to change bounds
	savePlot = input("Save plot? (Y/[N]): ")

	## If user wants to save
	if savePlot == 'Y':

		## Output path
		outPath = os.path.join("plots", outDir, outFile)

		## If directory for calibration curve outputs does not exist
		if not os.path.isdir("plots/"):

			# Make directory for calibration files
			os.mkdir("plots/")

		## If directory for output file does not exist
		if not os.path.isdir(os.path.join("plots", outDir)):

			# Make directory for output files
			os.mkdir(os.path.join("plots", outDir))

		## Check if the calibration curve already exists
		if len(glob.glob(outPath)) > 0:

			print(f'Plot {outFile} already exists! Delete or rename before running again.')

			print(f'\nPath to existing file: {outPath}')

			# Exit out of script
			return

		## Save plot
		plt.savefig(outPath, dpi=300)

	## Display plot
	plt.show()

	return

### Function to create csv file from spectrum data file and calibration curve
def createCsv(filePath, caliPath):

	## Load selected spectrum from filePath
	meta, spectrum = readSpectrumFile(filePath)

	## Obtain binning
	binning = np.arange(len(spectrum))

	## Load calibration curve
	caliDF = pd.read_csv(caliPath)

	## Calibration curve
	m = caliDF['m'][0]
	b = caliDF['b'][0]

	## Calibrate binning
	energies = m*binning + b

	## Create new dictionary with data
	df = {
		'energy (keV)': energies,
		'counts': spectrum,
	}

	## Convert dictionary to pandas dataframe
	df = pd.DataFrame(df)

	## Split filepath into individual componenets
	pathComps = filePath.split(os.sep)

	## Filename as csv
	outFile = pathComps[-1].split('.')[-2] + "_calibrated.csv"

	## File path to create in output
	outDir = pathComps[-2]

	## Output path
	outPath = os.path.join("csvOut", outDir, outFile)

	## If directory for csv outputs does not exist
	if not os.path.isdir("csvOut/"):

		# Make directory for csv files
		os.mkdir("csvOut/")

	## If directory for output file does not exist
	if not os.path.isdir(os.path.join("csvOut", outDir)):

		# Make directory for output files
		os.mkdir(os.path.join("csvOut", outDir))

	## Check if the calibration curve already exists
	if len(glob.glob(outPath)) > 0:

		print(f'Plot {outFile} already exists! Delete or rename before running again.')

		print(f'\nPath to existing file: {outPath}')

		# Exit out of script
		return

	## Save csv
	df.to_csv(f"{outPath}", index=False)

	print(f"Saved data to {outPath}.")

### Main functioning of script
def main(args):

	## Check if user wants to fit calibration datapoints
	if args.cPoints:

		print(f"Fitting calibration curve points from {args.src}.\n")

		# Run createCalibration
		calibrationPoints(args.src)

		return

	## Check if user wants to create a calibration file
	if args.calibrate:

		print(f"Creating a calibration curve.\n")

		# Run createCalibration
		createCalibration()

		return

	## If the user wants to display spectrum
	if args.display:

		print(f"Displaying {args.src} using {args.cSrc} for calibration.\n")

		# Run displaySpectrum
		displaySpectrum(args.src, args.cSrc)

		return

	## If the user wants to export data to a csv
	if args.csv:

		print(f"Creating csv file from {args.src} calibrated using {args.cSrc}.\n")

		# Run createCsv
		createCsv(args.src, args.cSrc)

		return

	## If nothing was specified
	else:

		print("Specify arguments to perform task.")

	return


### Run if file is run directly
if __name__ == '__main__':

	## Create new parser
	parser = argparse.ArgumentParser(description='Process inputs to calibrate spectra.')

	## Choose spectrum source
	parser.add_argument('--src', action='store', nargs='?', type=str, default='spectra/demo/2022_12_02_CdTe_Zn_01_no_purge.txt', help='Spectrum source file.')

	## Choose whether to create spectrum calibration file
	parser.add_argument('--cPoints', action='store_true', help='Choose whether to fit points to get calibration points.')

	## Choose whether to create spectrum calibration file
	parser.add_argument('--calibrate', action='store_true', help='Choose whether to create spectrum calibration file.')

	## Choose calibration source file
	parser.add_argument('--cSrc', action='store', nargs='?', type=str, default='cCurves/demo/2022_12_02_CdTe_Zn_01_no_purge_curve.csv', help='Spectrum source file.')

	## Choose whether to display a spectrum
	parser.add_argument('--display', action='store_true', help='Choose whether to display a spectrum.')

	## Choose whether to create a csv from spectrum
	parser.add_argument('--csv', action='store_true', help='Choose whether to create a csv from spectrum.')

	## Parse arguments
	args = parser.parse_args()

	## Call main
	main(args)

	## Example commands:

	# To obtain calibration points:
	# python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cPoints

	# To create a calibration curve from many points:
	# python main.py --calibrate

	# To display a spectrum:
	# python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cSrc cCurves\demo\2022_12_02_CdTe_Zn_01_no_purge_curve.csv --display

	# To create a csv of a spectrum:
	# python main.py --csv --src spectra\2022_09_30\2022_09_30_CdTe_Zn_02.txt --cSrc cCurves\EM_CdTe_Calibration_Curve_1\EM_CdTe_curve_1.csv

	## Metadata
	# Note: Data is stored in the STEAM shared Google Drive at .