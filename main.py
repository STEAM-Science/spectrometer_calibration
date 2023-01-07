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


### Create calibration for an element using known spectrum
def createCalibration(filePath):

	## Normalize filepath
	filePath = os.path.normpath(filePath)

	## Split filepath into individual componenets
	pathComps = filePath.split(os.sep)

	## Find element from filePath
	element = pathComps[-1].split('_')[4]

	## Filename as calibration curve
	outFile = pathComps[-1].split('.')[-2] + "_curve.csv"

	## File path to create in output
	outDir = pathComps[-2]

	## Output path
	outPath = os.path.join("cCurves", outDir, outFile)

	## If directory for calibration curve outputs does not exist
	if not os.path.isdir("cCurves/"):

		# Make directory for calibration files
		os.mkdir("cCurves/")

	## If directory for output directory does not exist
	if not os.path.isdir(os.path.join("cCurves", outDir)):

		# Make directory for output files
		os.mkdir(os.path.join("cCurves", outDir))

	## Check if the calibration curve already exists
	if len(glob.glob(outPath)) > 0:

		print(f'Calibration curve {outFile} already exists! Delete or rename before running again.')

		print(f'\nPath to existing file: {outPath}')

		# Exit out of script
		return

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
			'title': f'{element}: choose a peak and close window',
			'legend': True
		}
		
		## Plotting uncalibrated spectrum
		plotData(binning, spectrumUncalibrated, plotArgs)

		## Showing plot
		plt.show()

		## Get an input of peak index
		pNum = int(getNumericalInput('Input index of NIST peak to fit: '))

		## Obtain fit gaussian from spectrum
		A, sigma, mean = obtainPeak(binning, spectrumUncalibrated, element)

		## Get energy of fit peak
		pEnergy = NISTCoords[pNum-1, 0]

		## Add pair of calibration point to array with uncertainty
		calibration.append(np.array([pNum, pEnergy, sigma]))

	## Convert calibration points to array
	calibration = np.asarray(calibration)

	## Get list of 'x' and 'y' calibration points
	calibration = calibration.T

	## Get calibration curve
	m = m_weighted(calibration[0], calibration[1], calibration[2])
	b = b_weighted(calibration[0], calibration[1], calibration[2])
	mErr = sig_m_weighted(calibration[0], calibration[1], calibration[2])
	bErr = sig_b_weighted(calibration[0], calibration[1], calibration[2])

	print(f'\nFinal calibration curve: y = {m:.3g}x + {b:.3g}')
	print(f'Errors:')
	print(f'm = {m:.2f} ± {mErr:.2f}')
	print(f'b = {b:.2f} ± {bErr:.2f}')
			
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

	## Create pandas dataframe from dictionary
	df = pd.DataFrame.from_dict(calibration)

	## Save calibration data to csv
	df.to_csv(outPath, index=False, encoding='utf-8')

	print(f'\nSaved calibration curve to {outPath}.')

	return

### Display a spectrum
def displaySpectrum(filePath, caliPath, save=False):

	## Load selected spectrum from filePath
	meta, spectrumUncalibrated = readSpectrumFile(filePath)

	## Obtain binning
	binning = np.arange(len(spectrumUncalibrated))

	## Load calibration curve
	caliDF = pd.read_csv(caliPath)

	## Calibration curve
	m = caliDF['m']
	b = caliDF['b']

	## Calibrate binning
	energies = m*binning + b

	## Find element from filePath
	element = pathComps[-1].split('_')[4]

	## Create dictionary to store plotting parameters
	plotArgs = {
		'color': 'k',
		'label': 'data',
		'xlabel': 'bin',
		'ylabel': 'counts',
		'title': f'{element}: spectrum',
		'legend': True
	}

	## Plot data
	plotData(binning, spectrumUncalibrated, plotArgs)

	return

### Function to create csv file from spectrum data file and calibration curve
def createCsv(filePath, caliPath):

	## Use readSpectrumFile() to read file
	calibrationData, spectralData = readSpectrumFile(filePath)


	### INCOMPLETE

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

		print(f"Displaying {args.src} using {args.cSrc} for calibration.\n")

		# Run displaySpectrum
		displaySpectrum(args.src, args.cSrc, save=args.save)

		return

	## If the user wants to export data to a csv
	if args.csv:

		print(f"Creating csv file from {args.src}.\n")

		# Run createCsv
		createCsv(args.src)

		return

	## If nothing was specified
	else:

		print("Specify arguments to perform task.")

	return


### Run if file is run directly
if __name__ == '__main__':

	## Create new parser
	parser = argparse.ArgumentParser(description='Process inputs to calibrate spectra.')

	## Choose spectrum sourse
	parser.add_argument('--src', action='store', nargs='?', type=str, default='spectra/demo/2022_12_02_CdTe_Zn_01_no_purge.txt', help='Spectrum source file.')

	## Choose whether to create spectrum calibration file
	parser.add_argument('--calibrate', action='store_true', help='Choose whether to create spectrum calibration file.')

	## Choose calibration source file
	parser.add_argument('--cSrc', action='store', nargs='?', type=str, default='cCurves/Ba_2022_11_30_curve_01.csv', help='Spectrum source file.')

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

	## Example commands:

	# To calibrate a spectrum:
	# python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --calibrate

	# To display a spectrum: (Add '--save' at the end to save the displayed spectrum.)
	# python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cSrc cCurves/Ba_2022_11_30_curve_01.csv --display

	# To create a csv of a spectrum:
	# python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cSrc cCurves/Ba_2022_11_30_curve_01.csv --csv

	## Metadata
	# Note: Data is stored in the STEAM Google Drive at .