import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


import utils.spectrum as spectrum
import utils.files as files
import utils.gaussian_fit as gauss
import utils.plot as plot
import utils.llsfit as fit

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

def calibration_points():
	"""
	Prompt the user to select regions of interest in a spectrum and fit Gaussian curves to them.
	The parameters from the Gaussian fit are saved as a CSV file in the 'cPoints' directory.
	If the calibration curve already exists, the function exits without performing any operation.

	Args:
		filePath (str): The path of the file containing the spectrum data.
	"""

	user_input = input("What element is being calibrated?: ")
	element = user_input.capitalize()

	## Load selected file(s)
	spectral_data = spectrum.process_spectrum()

	num_items_to_process = len(spectral_data)
	total_items_processed = 0

	folder_path = files.get_folder_path()

	## For every spectrum in the list
	for data_processed in spectral_data:
		
		try:
			data = data_processed.data
			print("\nGetting calibration points...")
			
			
			## Obtain binning
			binning = np.arange(len(data))

			## TODO: NIST PATH HERE
			## Filepath to corresponding NIST spectrum
			root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))			
			nist_path = f'{root_dir}/nist_data/Spectral_Lines_{element}.csv'
			

			## Read NIST data
			nist_x, nist_y = np.loadtxt(nist_path, delimiter=',').T

			# Create mask to clip data to below 50 keV
			mask = nist_x < 50
			nist_x = nist_x[mask]
			nist_y = nist_y[mask]

			# Recompose into matrix of coordinates
			nist_coords = np.array([nist_x, nist_y]).T

			## Display spectrum (uncalibrated)
			print("\nDisplaying uncalibrated spectrum. Select how many peaks to fit")

			## Plot parameters
			plotArgs = {
				'color': 'k',
				'label': 'data',
				'xlabel': 'bin',
				'ylabel': 'counts',
				'title': f'{element}: count peaks that will be fit and close window',
				'legend': True
				}

			# Check how many bins the spectrum contains
			nBins = len(data)

			# Clipping noise (about bin 100 or 1/10 of the spectrum)
			cutoff = int(nBins/10)

			# Find the largest value in the spectrum after cuffoff
			clipVal = np.max(data[cutoff:])*1.5

			# Clip the spectrum to remove noise
			spectrum_temp = np.clip(data, a_min = 0, a_max = clipVal)

			## Plot and display the spectrum
			plot.plot_data(binning, spectrum_temp, plotArgs)
			plt.show()

			## Get input points from user (click 2 points on the display)
			num_peaks = 0
			while num_peaks < 1:
				num_peaks = int(getNumericalInput("Input number of peaks to fit: "))

			## Create list to store calibration points
			points = []

			# Iterator
			nPeak = 1

			# For every peak in the range
			while nPeak < num_peaks+1:
				
				print(f"Select peak {nPeak} (or {nPeak} and {nPeak+1} for a doublet) from {element} spectrum")

				plot.nist_data(nist_coords, units='keV')

				print("\nDisplaying uncalibrated spectrum. Close once a peak has been choosen.")

				## Plot parameters
				plotArgs = {
					'color': 'k',
					'label': 'data',
					'xlabel': 'bin',
					'ylabel': 'counts',
					'title': f'{element}: choose a peak and close window',
					'legend': True
					}

				# Plot spectrum
				plot.plot_data(binning, spectrum_temp, plotArgs)
				plt.show()

				## Get input of peak index
				peak_index = int(getNumericalInput("Input index of NIST peak fo fit (enter 999 for double peak): "))

				# Check if single peak
				if peak_index != 999:

					# Obtain fit gaussian parameters
					gauss_params, params_covariance, int_counts, max_counts = plot.select_peak(binning, data, element)

					# Unpacking parameters
					A, sigma, mean = gauss_params

					# Unpacking errors
					A_err = np.sqrt(params_covariance[0,0]) 
					sigma_err = np.sqrt(params_covariance[1,1])
					mean_err = np.sqrt(params_covariance[2,2])

					# Get energy of peak
					peak_energy = nist_coords[peak_index-1, 0]

					# Add pair of calibration point to array with uncertainties
					points.append(np.array([mean, mean_err, peak_energy, sigma, sigma_err, A, A_err, int_counts, max_counts]))

					# Iterate
					nPeak += 1

				# Double peak
				else:
					print("\nFitting a Gaussian doublet. Ensure you go from left to right!")

					# Get peak indexes
					peak_1 = int(getNumericalInput('Input index of lower energy NIST peak in doublet:  '))
					peak_2 = int(getNumericalInput('Input index of higher energy NIST peak in doublet: '))

					print("\nFitting first peak in doublet: ")

					# Obtain fit gaussian parameters
					gauss_params_1, params_covariance_1, int_counts_1, x_coords_1, max_counts_1 = plot.select_peak(binning, data, element, doublet=True)
					gauss_params_2, params_covariance_2, int_counts_2, x_coords_2, max_counts_2 = plot.select_peak(binning, data, element, doublet=True)

					# Temporary variables
					temp1_1, temp2_1, temp3_1 = gauss_params_1
					temp1_2, temp2_2, temp3_2 = gauss_params_2

					# Obtain predicted fit parameters for double gaussian
					p0_double = np.array([temp1_1, temp2_1, temp3_1, temp1_2, temp2_2, temp3_2])

					# Create mask for sliciing
					mask = (binning > x_coords_1[0])*(binning < x_coords_2[-1])

					# Slicking spectrum to fit double gaussian
					binning_slice = binning[mask]
					spectrum_slice = data[mask]

					# Compute double gaussian fit
					gauss_params, params_covariance = curve_fit(gauss.double_gaussian, binning_slice, spectrum_slice, p0=p0_double)

					# Estimate integrated counts from gaussian
					int_counts_1 = np.sum(data[int(x_coords_1[0]):int(x_coords_1[-1])])
					int_counts_2 = np.sum(data[int(x_coords_2[0]):int(x_coords_2[-1])])

					print('\nEstimated fit parameters (peak 1):')
					print('A1 = {:.7g}'.format(gauss_params[0]))
					print('σ1 = {:.7g}'.format(gauss_params[1]))
					print('μ1 = {:.7g}'.format(gauss_params[2]))

					print('\nEstimated fit parameters (peak 2):')
					print('A2 = {:.7g}'.format(gauss_params[3]))
					print('σ2 = {:.7g}'.format(gauss_params[4]))
					print('μ2 = {:.7g}'.format(gauss_params[5]))

					# Unpacking fit parameters
					A_1, sigma_1, mean_1, A_2, sigma_2, mean_2 = gauss_params

					# Create dictionary to store plotting parameters
					plotArgs = {
						'color': 'k',
						'label': 'data',
						'xlabel': 'MCA bin',
						'ylabel': 'counts',
						'title': f'{element}: fitted double gaussian',
						'legend': True
					}

					# Plot data from spectrum
					plot.plot_data(binning_slice, spectrum_slice, plotArgs, xBounds=(int(x_coords_1[0]), int(x_coords_2[-1])))

					# Plot fit spectrum
					plt.plot(binning_slice, gauss.double_gaussian(binning_slice, *gauss_params), label='fit')

					plt.show()

					# Unpacking errors on fit parameters
					A_err_1 = np.sqrt(params_covariance[0,0])
					sigma_err_1 = np.sqrt(params_covariance[1,1])
					mean_err_1 = np.sqrt(params_covariance[2,2])
					A_err_2 = np.sqrt(params_covariance[3,3])
					sigma_err_2 = np.sqrt(params_covariance[4,4])
					mean_err_2 = np.sqrt(params_covariance[5,5])

					# Get energies of fit peaks
					peak_energy_1 = nist_coords[peak_1-1, 0]
					peak_energy_2 = nist_coords[peak_2-1, 0]

					# Add pair of calibration point to array with uncertainties
					points.append(np.array([mean_1, mean_err_1, peak_energy_1, sigma_1, sigma_err_1, A_1, A_err_1, int_counts_1, max_counts_1]))
					points.append(np.array([mean_2, mean_err_2, peak_energy_2, sigma_2, sigma_err_2, A_2, A_err_2, int_counts_2, max_counts_2]))

					# Iterate
					nPeak += 2
			
			## Convert calibration points to array
			points = np.asarray(points)

			## Get list of 'x' and 'y' calibration points
			points = points.T

			pointsDict = {
				'mu': points[0],
				'muErr': points[1],
				'muRErr': points[1]/points[0],
				'E': points[2],
				'sigma': points[3],
				'sigmaErr': points[4],
				'sigmaRErr': points[4]/points[3],
				'A': points[5],
				'AErr': points[6],
				'ARErr': points[6]/points[5],
				'intCounts': points[7],
				'max_counts': points[8]
			}

			## Create pandas dataframe from dictionary
			df = pd.DataFrame.from_dict(pointsDict)

			## Name of csv (Original filename + _points)
			file_name = data_processed.filename + '_points'
			
			## Save dataframe to CSV file
			files.create_csv(df, file_name, folder_path)

			print("Calibration points saved to CSV file.")

			total_items_processed += 1

		except Exception as e:
			print(f"Error processing {data_processed.filename}")
			print(f"Error Message: {e}")
			print("Please reprocess this file.")
			print("Skipping file....")

		finally:
			print(f"{total_items_processed} of {num_items_to_process} files processed.")

	return

def create_calibration_curve():
	"""
	Creates a calibration curve from a set of input files containing calibration points.

	The function prompts the user to input the file paths for each set of calibration points.
	It then calculates the calibration curve and saves the results to a CSV file.
	Additionally, it saves a plot of the calibration curve and displays it to the user.
	"""

	print("Creating calibration curve...")

	user_input = input("Do you want to process a folder (f) or individual files (i)?: ")
	
	# Removing case sensitivity of user input
	user_input_case = user_input.lower()
	
	if user_input_case == "f":
		calibration_points = files.load_folder()
	elif user_input_case == "i":
		calibration_points = files.load_files()
	else: 
		raise ValueError("Invalid input")
	
	print(len(calibration_points), "files loaded\n")

	## Creating arrays to store calibration points
	mus = np.array([])
	Es = np.array([])
	sigmas = np.array([])

	## Getting calibration points from each file
	for data in calibration_points:

		df = pd.read_csv(data, index_col=False)

		mus = np.concatenate((mus, df['mu']))
		Es = np.concatenate((Es, df['E']))
		sigmas = np.concatenate((sigmas, df['muRErr']))
	
	## Get calibration curve
	m = fit.m_weighted(mus, Es, sigmas)
	b = fit.b_weighted(mus, Es, sigmas)
	m_err = fit.sig_m_weighted(mus, Es, sigmas)
	b_err = fit.sig_b_weighted(mus, Es, sigmas)

	## Calculating R-squared
	# Model predicted energies
	model_Es = m*mus + b

	# R-squared it then;
	R_sq = 	1 - (np.sum((Es - model_Es)**2)/np.sum((Es - np.mean(Es))**2))

	print(f'R-squared = {R_sq:.4f}')

	## Creating dictionary to store calibration curve values
	calibration = {
		'm': np.array([m]),
		'b': np.array([b]),
		'mErr': np.array([m_err]),
		'bErr': np.array([b_err]),
		'r-squared': np.array([R_sq])
	}

	# Converting dictionary to pandas dataframe
	df = pd.DataFrame.from_dict(calibration)

	## Saving calibration curve to CSV file
	name = input("\nInput name of calibration curve: ")

	print("\nSaving calibration curve to CSV file...")
	files.create_csv('curve', df, name)
	print("\nSaving loaded files calibration points as a CSV file...")
	#files.create_csv()

	print("\nSaving complete....")
	print("\nDisplaying calibration curve...")

	## Plot calibration curve
	plt.clf()

	plt.scatter(mus, Es, label='datapoints', color='k', zorder=2)

	# Points to dislpay fit
	x_fit = np.linspace(0, 1024, num=int(1e3))
	E_fit = m * x_fit + b

	# Plot
	plt.plot(x_fit, E_fit, label=f'fit (y = {m:.5f}x + {b:.5f})', color='b', zorder=1)

	# Plot properties
	plt.xlabel('MCA channel')
	plt.ylabel('Energy (keV)')
	plt.xlim(0, 1023)
	plt.legend()

	## Save figure
	#plt.savefig(f'../calibration_plots/{name}.png', dpi=300)

	## Display plot
	plt.show()

	print(f"Calibration curve saved as {name}_plot.png.")

	return