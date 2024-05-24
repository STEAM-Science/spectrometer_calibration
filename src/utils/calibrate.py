"""
calibrate.py - A module for calibrating spectral data.

This module provides functions for calibrating spectral data using NIST calibration points.

Functions:
- calibration_points(): Performs a Gaussian fit over a selected region of interest and saves the results to a CSV file.
- create_calibration_curve(): Creates a calibration curve from a set of input files containing calibration points. 
- determine_resolution(): Determines the resolution of the spectrometer from a set of input files containing calibration results.
- determine_response(): Determines the response of the spectrometer from a set of input files containing calibration results.
- calibrate_spectrum(): Calibrates a spectrum using a calibration curve.

"""

### Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

### Importing functions from other python files in the project
import utils.spectrum as spectrum
import utils.files as files
import utils.gaussian_fit as gauss
import utils.plot as plot
import utils.llsfit as fit
import utils.classes as classes
import utils.simulated as sim

## General functions
def get_isotopes():
	"""
	Prompts the user to input isotopes to fit and returns a list of the entered isotopes
	
	Returns:
		user_isotopes (list): A list of the entered isotopes.
	"""
	print("For the following prompt, please put each isotope used one at a time. Only include the mass number (ex: 133 for Cs-133) if necessary. Enter the same format style as your data (ex: Cs-133, Cs_133, or Cs133)")

	user_isotopes = []

	user_input = input("Input isotopes to fit (leave blank to quit): ")

	while user_input != "":
		user_isotopes.append(user_input)
		user_input = input("Input isotopes to fit (leave blank to quit): ")
	
	print("You entered the following isotopes:", ', '.join(user_isotopes))
	
	return user_isotopes

def getNumericalInput(text):
	"""
	Prompts the user to input a numerical value and returns it as a float.

	Args:
		text (str): user input.

	Returns:
		float: The numerical value inputted by the user.

	Raises:
		ValueError: If the user inputs a non-numerical value.
	"""

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

### Calibration functions

def calibration_points():
	"""
	Select regions of interest in a spectrum and fit Gaussian curves to them.
	The results from the Gaussian fit are saved as a CSV file.

	Step 1. Choose a folders or files containing calibration points 
	Step 2. Read the spectrum data from the file(s) 
	Step 3. Plot the spectrum data and select region of interest 
	Step 4. Perform a Gaussian fit over the selected region 
	Step 5. Save the Gaussian fit parameters to an array
	Step 6. Repeat steps 3-5 for each region of interest
	Step 7. Save the Gaussian fit parameters to a csv file
	"""

	## Step 1. Choose a folders or files containing calibration points using the function spectrum.process_spectrum()

	# Load selected file(s) using functions from spectrum.py
	spectral_data = spectrum.process_spectrum()

	# Number of files to process
	num_items_to_process = len(spectral_data)

	# Iterator to track progress
	total_items_processed = 0

	# Get folder path from user (where results will be saved)
	folder_path = files.get_folder_path()

	## Step 2. Read the spectrum data from the file(s) 
	
	# For every spectrum in the list
	for data_processed in spectral_data:
		
		# Iterating through all files in the list may raise an error or exception. To ensure the user does not have
		# to start over, the program will continue to iterate through the list even if an error is raised.
		try:
			## Asks user which element is being calibrated
			user_input = input("What element is being calibrated?: ")
			element = user_input.capitalize()

			# Obtain data file(s) from list (stored in a class)
			data = data_processed.data
			print("\nGetting calibration points...")
			
			## Obtain binning
			binning = np.arange(len(data))

			## Filepath to corresponding NIST spectrum, this will be displayed later for the user to compare
			## their spectrum to the NIST spectrum

			# Get root directory
			root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	

			# Find NIST data		
			nist_path = f'{root_dir}/nist_data/Spectral_Lines_{element}.csv'
			
			# Read NIST data
			nist_x, nist_y = np.loadtxt(nist_path, delimiter=',').T

			# Create mask to clip data to below 50 keV
			mask = nist_x < 50
			nist_x = nist_x[mask]
			nist_y = nist_y[mask]

			# Recompose into matrix of coordinates
			nist_coords = np.array([nist_x, nist_y]).T

			## Step 3. Plot the spectrum data and select region of interest
			# Display spectrum (uncalibrated)
			print("\nDisplaying uncalibrated spectrum. Select how many peaks to fit")

			# Plot parameters
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

			# Plot and display the spectrum
			plot.plot_data(binning, spectrum_temp, plotArgs)
			plt.show()

			## Allow user to select regions of interest
			# Get input points from user (click 2 points on the display)
			num_peaks = 0
			while num_peaks < 1:
				num_peaks = int(getNumericalInput("Input number of peaks to fit: "))

			## Create list to store calibration points
			points = []

			# Iterator
			nPeak = 1

			## Step 4. Perform a Gaussian fit over the selected region 

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

				## Get input of peak index from NIST data
				peak_index = int(getNumericalInput("Input index of NIST peak fo fit (enter 999 for double peak): "))

				# Check if single peak
				if peak_index != 999:

					## Obtain region of interest from user using select_peak function from plot.py 

					# Obtain fit gaussian parameters
					gauss_params, params_covariance, int_counts, maxCounts = plot.select_peak(binning, data, element)

					# Unpacking parameters
					A, sigma, mean = gauss_params

					# Unpacking errors
					A_err = np.sqrt(params_covariance[0,0]) 
					sigma_err = np.sqrt(params_covariance[1,1])
					mean_err = np.sqrt(params_covariance[2,2])

					# Get energy of peak
					peak_energy = nist_coords[peak_index-1, 0]

					## Step 5. Save the Gaussian fit parameters to an array (for single Gaussian)

					# Add pair of calibration point to array with uncertainties
					points.append(np.array([mean, mean_err, peak_energy, sigma, sigma_err, A, A_err, int_counts, maxCounts]))

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
					gauss_params_1, params_covariance_1, int_counts_1, x_coords_1, maxCounts_1 = plot.select_peak(binning, data, element, doublet=True)
					gauss_params_2, params_covariance_2, int_counts_2, x_coords_2, maxCounts_2 = plot.select_peak(binning, data, element, doublet=True)

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

					# Compute double gaussian fit using double_gaussian function from gaussian_fit.py
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
						'title': f'{element}: fitted double gaussian. Close to continue',
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

					## 	Step 5. Save the Gaussian fit parameters to an array (for double Gaussian)

					# Add pair of calibration point to array with uncertainties
					points.append(np.array([mean_1, mean_err_1, peak_energy_1, sigma_1, sigma_err_1, A_1, A_err_1, int_counts_1, maxCounts_1]))
					points.append(np.array([mean_2, mean_err_2, peak_energy_2, sigma_2, sigma_err_2, A_2, A_err_2, int_counts_2, maxCounts_2]))

					# Iterate
					nPeak += 2
			
			## 	Step 7. Save the Gaussian fit parameters to a csv file

			# Convert calibration points to array
			points = np.asarray(points)

			# Get list of 'x' and 'y' calibration points
			points = points.T

			points_dict = {
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
				'maxCounts': points[8]
			}

			# Create pandas dataframe from dictionary
			df = pd.DataFrame.from_dict(points_dict)

			# Name of csv (Original filename + _points)
			file_name = data_processed.filename + '_points'
			
			# Save dataframe to CSV file using create_csv function in files.py
			files.create_csv(df, file_name, folder_path)

			print("Calibration points saved to CSV file.")

			## Iterator
			total_items_processed += 1

		## If an error is raised, the program will continue to iterate through the list of files
		except Exception as e:

			# Tells user which file caused the error
			print(f"Error processing {data_processed.filename}")
			print("Please reprocess this file.")
			print("Skipping file....")

			# If there is a different error, tell user error. Ex: numpy error
			print(f"Error Message: {e}")

		## This block will always execute
		finally:

			# Print progress
			print(f"{total_items_processed} of {num_items_to_process} files processed.")

	return

def create_calibration_curve():
	"""
	Creates a calibration curve from a set of input files containing calibration points.

	Step 1: Get calibration points from each file and put data into the IsotopeCalibrationData class
	Step 2: Combine data from all files
	Step 3: Perform weighted fit using the combined data to get m and b
	Step 4: Plot the combined data using x=E, y=mu and create a legend for each isotope
	Step 5: Plot fit using m and b 
	Step 6: Display and save plot as png.
	Step 7: Save combine data and fit results as csv file
	"""

	## Step 1: Get calibration points from each file
	# Getting which isotopes were used in the calibration from user
	isotopes = get_isotopes()

	user_input = input("Do you want to process a folder (f) or individual files (i)?: ")
	
	# Removing case sensitivity of user input
	user_input_case = user_input.lower()
	
	# Load files or folder(s) based on user input
	if user_input_case == "f":
		calibration_points_files = files.load_folder()
	elif user_input_case == "i":
		calibration_points_files = files.load_files()
	else: 
		raise ValueError("Invalid input")
	
	## Step 2: Combine data from all files

	# Iterating through all the files in the list may raise an error or exception. 
	# If so, the program will stop and raise the error.
	try:

		# Initiating the CombinedIsotopeCalibrationData class from classes.py
		combined_data = classes.CombinedIsotopeCalibrationData()

		# Iterating through all the files in the list
		for file in calibration_points_files:
			for isotope in isotopes:

				# Checking if the isotope is in the file name
				if isotope in file.stem:

					# Reads csv file as a pandas dataframe
					df = pd.read_csv(file)

					# Unpacking data
					mu = df['mu']
					E = df['E']
					sigma = df['sigma']
					int_counts = df['intCounts']
					max_counts = df['maxCounts']
					
					# Combines arrays from all files and stores into class. For format information, see classes.py
					combined_data.add_mus(isotope, mu)
					combined_data.add_Es(isotope, E)
					combined_data.add_sigmas(isotope, sigma)
					combined_data.add_int_counts(isotope, int_counts)
					combined_data.add_maxCounts(isotope, max_counts)

		## Step 3: Perform weighted fit using the combined data to get m and b

		# Getting weighted fit parameters from class
		mu_values = [x[1] for x in combined_data.mu]
		E_values = [x[1] for x in combined_data.E]
		sigma_values = [x[1] for x in combined_data.sigma]

		# Converting lists to arrays
		mu_array = np.concatenate(np.array(mu_values, dtype=object))
		E_array = np.concatenate(np.array(E_values, dtype=object))
		sigma_array = np.concatenate(np.array(sigma_values, dtype=object))

		# Performing weighted fit using functions from llsfit.py
		m = fit.m_weighted(mu_array, E_array, sigma_array)
		b = fit.b_weighted(mu_array, E_array, sigma_array)
		m_err = fit.sig_m_weighted(mu_array, E_array, sigma_array)
		b_err = fit.sig_b_weighted(mu_array, E_array, sigma_array)
			
		# Model predicted energies
		model_Es = m*mu_array + b

		# Calculating R-squared
		R_sq = 	1 - (np.sum((E_array - model_Es)**2)/np.sum((E_array - np.mean(E_array))**2))

		# Printing results
		print(f'R-squared = {R_sq:.4f}')

		## From user, getting name of plot/files and folder path to save
		name = input("Input name of calibration curve plot: ")
		folder_path = files.get_folder_path()

		
		# Step 4: Plot the combined data using x=E, y=mu and create a legend for each isotope

		plt.clf()

		# Define markers and colors
		#markers = ['o', 's', '^', 'v', '*']
		#colors = ['yellow', 'tab:purple', 'r', 'b', 'g']

		#isotope_styles = {isotope: {'marker': marker, 'color': color} 
		 			#	for isotope, marker, color in zip(isotopes, markers, colors)}
		
		# Create a dictionary mapping isotopes to markers and colors
		isotope_styles = {
				'Am': {'marker': 'o', 'color': 'yellow'},
				'Ba': {'marker': 's', 'color': 'tab:purple'},
				'Fe': {'marker': 'v', 'color': 'r'},
				'Zn': {'marker': '^', 'color': 'b'},
				'Cd': {'marker': '*', 'color': 'g'}
			}
		
		plt.figure()
		# Iterate through all isotopes
		for isotope in isotopes:

			# Iterate through all data in combined_data to separate mu and E data for each isotope
			for y in combined_data.mu:
				for x in combined_data.E:
					if x[0] == isotope and y[0] == isotope:

						# Unpacking data
						isotope_E_data = x[1]
						isotope_mu_data = y[1]

						#plot.isotope_styles(isotope_mu_data, isotope_E_data, isotope)

						# Get the marker and color for the current isotope
						marker = isotope_styles[isotope]['marker']
						color = isotope_styles[isotope]['color']

						# Plot the data for the current isotope
						plt.scatter(isotope_mu_data, isotope_E_data, label=isotope, zorder=2, marker=marker, color=color, edgecolors='k')

		## Step 5: Plot fit using m and b
		# Points to display calibration curve fit
		x_fit = np.linspace(0, 1024, num=int(1e3))
		E_fit = m * x_fit + b

		# Plot fit
		plt.plot(x_fit, E_fit, label=f'fit (y = {m:.5f}x + {b:.5f})', color='k', zorder=1)

		## Step 6: Display and save plot as png.
		# Set plot properties
		#plt.title(f'{name} Calibration Curve')
		plt.xlabel('MCA channel')
		plt.ylabel('Energy (keV)')
		plt.xlim(0, 1023)
		plt.legend()

		# Save figure using create_image function in files.py
		files.create_image(name, folder_path)

		# Display the plot
		plt.show()

		plt.figure()
		# Iterate through all isotopes
		for isotope in isotopes:

			# Iterate through all data in combined_data to separate mu and E data for each isotope
			for y in combined_data.mu:
				for x in combined_data.E:
					if x[0] == isotope and y[0] == isotope:

						# Unpacking data
						isotope_E_data = x[1]
						isotope_mu_data = y[1]

						#plot.isotope_styles(isotope_mu_data, isotope_E_data, isotope)

						# Get the marker and color for the current isotope
						marker = isotope_styles[isotope]['marker']
						color = isotope_styles[isotope]['color']
		
						## Plotting residuals
						# Residuals
						# Energy - (m*counts + b)
						residuals = isotope_E_data - (m*isotope_mu_data + b)

						residual_errors = np.std(residuals)

						plt.errorbar(range(len(residuals)), residuals, yerr=residual_errors, fmt='o')
		
			#plt.title(f'{name} Calibration Curve Residuals')
			plt.axhline(0, color='k', linestyle='--')
			plt.xlabel('MCA channel')
			plt.ylabel('Residuals')
			plt.xlim(0, 1023)
			plt.legend()

			files.create_image(name + '_residuals', folder_path)

			plt.show()



		## Step 7: Save combined data and fit results as csv
		# Creating dictionary to store combined data, without isotope information
		combined_dict = {
			'mu': mu_array,
			'E': E_array,
			'sigma': sigma_array
			#'int_counts': int_counts_array,
			#max_counts = max_counts_array
		}

		# Creating dictionary to store fit results
		calibration_dict = {
			'm': np.array([m]),
			'b': np.array([b]),
			'm_err': np.array([m_err]),
			'b_err': np.array([b_err]),
			'R_sq': np.array([R_sq])
		}
		
		# Create pandas dataframe from dictionary
		combined_df = pd.DataFrame.from_dict(combined_dict)
		calibration_df = pd.DataFrame.from_dict(calibration_dict, orient='index', columns=['value'])
		
		# Save dataframe to CSV file using create_csv function in files.py
		csv_combined = files.create_csv(combined_df, name +'_points', folder_path)
		csv_calibration = files.create_csv(calibration_df, name, folder_path)

		### End of function
		print("\nCalibration complete.")

		### Asking user if they wish to proceed to the next calibration step
		advance = input("Would you like to continue to the resolution (r) or response (rp) analysis? Leave blank to exit: ")

		# Removing case sensitivity of user input
		advance_case = advance.lower()

		## If user wishes to continue to the next step, call the next function (determine_resolution)
		while advance_case != "":

			# Go to determine_resolution
			if advance_case == "r":
				return #determine_resolution(advance_case, calibration_points)
			elif advance_case == "rp":
				return determine_response(advance_case, calibration_df)
			else:
				raise ValueError("Invalid input")
		
	except Exception as e:
		print(f'Error Message: {e}') 

def determine_response(advance=None, calibration_points=None):
	"""
	Determines the detector response by analyzing the Full Width Half Max (FWHM) and Energy (E) of the calibration points.
	Saves the detector response data to a CSV file and displays a plot of FWHM vs E.

	Step 1. Load a file(s) containing Gaussian fit parameters
	Step 2. Select sigmas from data 
	Step 3. Plot sigmas vs energy
	Step 4. Save the plot to an image file
	Step 5. Save the resolution to a csv file
	"""

	## Step 1. Load a file(s) containing Gaussian fit parameters

	# Either continuning from previous function or starting from scratch
	# If starting from scratch, uploading new file
	if advance == None:

		# Getting user input for calibration curve file path
		print("Only upload one file containing calibration curve points")
		user_input = input("Calibration curve points file path: ")

		# Removing quotes from user input
		user_input = user_input.replace('"', '')

		# Obtain data
		calibration_points = pd.read_csv(user_input, index_col=False) 
	
	# Continuing from previous function (create_calibration_curve or determine_resolution)
	if advance == "rp":
		pass
		

	print("Beginning detector response analysis...\n")
	
	## Step 2. Select sigmas from data and determine response Full Width Half Max (FWHM)
	# Calculating Full Width Half Max (FWHM)
	fwhm = calibration_points['sigma'] * 2.355
	E = calibration_points['E']

	## Creating a dictionary to store detector response data
	response_dict = {
		'E (keV)': E,
		'FWHM (keV)': fwhm
	}

	## Step 3. Plot sigmas vs energy
	# Plotting detector response: FWHM vs E
	plt.scatter(E, fwhm)
	
	# Getting user_input for name of plot and folder path
	name = input("Input name of resolution plot: ")
	folder_path = files.get_folder_path()

	# Plot properties
	plt.title(f'{name} Response')
	plt.xlabel('Energy (keV)')
	plt.ylabel('FWHM (keV)')

	## Step 4. Save the plot to an image file
	# Save figure using create_image function in files.py
	files.create_image(name, folder_path)

	# Display plot
	print("\nDisplaying detector response...")
	plt.show()

	## Step 5. Save the resolution to a csv file
	# Save detector response data to CSV file using create_csv function in files.py
	files.create_csv(response_dict, name, folder_path)
	
	print("\nComplete.")


	### Asking user if they wish to proceed to the next calibration step
	advance = input("Would you like to continue to the resolution analysis? y/n (Leave blank to exit): ")

	# Removing case sensitivity of user input
	advance_case = advance.lower()

	# If user wishes to continue to the next step, call the next function (determine_resolution)
	while advance_case != "":

		# Go to determine_resolution
		if advance_case == "y":
			return determine_resolution(advance_case, calibration_points)
		else:
			raise ValueError("Invalid input")
	return


def determine_resolution(advance=None, calibration_points=None):
	
## Here is a thought, could do something like i did for the thermal stuff, where they put all the stuff in the beginning
## and then it goes through the whole process.

	## Either continuning from previous function or starting from scratch
	# Starting from scratch, uploading new file
	if advance == None:
		# Asking user for calibration curve file path
		print("Please only upload one file containing calibration curve points")
		user_input = input("Calibration curve points file path: ")
		user_input = user_input.replace('"', '')

		# Obtaining data from file
		calibration_points = pd.read_csv(user_input, index_col=False)

		
	
	# Continuing from previous function (create_calibration_curve or determine_response)
	if advance == "r":
		pass

	
	# Checking if the user wants to run a simulated spectrum or if they have the data already
	simulated_input = input("Would you like to run a simulated spectrum? (y/n): ")

	# Simulated spectrum using simulated.py functions
	if simulated_input == "y":
		expected_spectra = sim.smooth_isotope_spectrum()
		#output_spectrum = {'element': element, 'energy': eee2, 'spectrum': smooth_spectrum_rebinned_noise}
	
	# Using expected spectra file (uploaded by user)
	elif simulated_input == "n":
		element = input("Please enter the element you are analyzing. Ex: Cs, Ba, Am, etc.: ")
		expected_spectra_input = input("Expected spectrum file path: ")
		expected_spectra_input = expected_spectra_input.replace('"', '')

		# Obtaining data from file
		expected_spectra = pd.read_csv(expected_spectra_input, index_col=False)

	
	print("Beginning resolution analysis...\n")

	element_dict = {
		'Am': [
			{'peak': 1, 'energy': 13.96, 'rel_counts': 9.6},
			{'peak': 2, 'energy': 17.751, 'rel_counts': 5.7},
			{'peak': 3, 'energy': 16.816, 'rel_counts': 2.5}
		],
		'Ba': [
			{'peak': 1, 'energy': 4.286, 'rel_counts': 6},
			{'peak': 2, 'energy': 4.717, 'rel_counts': 0.93},
		],
		'Fe': [
			{'peak': 1, 'energy': 5.899, 'rel_counts': 16.9},
			{'peak': 2, 'energy': 6.49, 'rel_counts': 1.98}
		],
		'Zn': [
			{'peak': 1, 'energy': 8.048, 'rel_counts': 23.4},
			{'peak': 2, 'energy': 8.905, 'rel_counts': 2.78}
		],
	}

	peaks = element_dict.get(element)

	# If the element exists in the dictionary
	if peaks is not None:
		for peak in peaks:
			energy = peak['energy']

			# Get the expected_count and max_count for this energy
			expected_counts = expected_spectra[expected_spectra['energy'] == energy]['counts']
			measured_counts = calibration_points[calibration_points['E'] == energy]['maxCounts']

			if expected_counts is not None:
				peak['expected_counts'] = expected_counts
			if measured_counts is not None:
				peak['measured_counts'] = measured_counts

	

	plt.plot(measured_counts, expected_counts)
	plt.show()

	


def calibrate_spectrum():
	'''
	Calibrates spectral data using a calibration curve and saves the calibrated data as a CSV file and image.
	
	Step 1. Upload a spectrum file and calibration curve file
	Step 2. Calibrate the spectrum using m and b from the calibration curve
	Step 3. Display the calibrated spectrum
	Step 4. Save the calibrated spectrum as a CSV file and image
	'''

	## Step 1. Upload a spectrum file and calibration curve file

	# Load calibration curve using functions from files.py
	user_input_calibration = input("\nEnter path to calibration curve file: ")

	calibration_curve = pd.read_csv(user_input_calibration)
	m = calibration_curve['m'][0]
	b = calibration_curve['b'][0]


	# Load selected file(s) to calibrate using functions from spectrum.py
	print("\nSelect which spectra to calibrate...")
	spectral_data = spectrum.process_spectrum()

	# Number of files to process
	num_items_to_process = len(spectral_data)

	# Iterator o keep track of progress
	total_items_processed = 0

	# Get folder path from user (where results will be saved)
	folder_path = files.get_folder_path()


	## Step 2. Calibrate the spectrum

	# For every spectrum in the list
	for data_processed in spectral_data:

		# Iterating through all files in the list may raise an error or exception. To ensure the user does not have
		# to start over, the program will continue to iterate through the list even if an error is raised.
		try:

			# Obtain data file(s) from list (stored in a class)
			data = data_processed.data

			# Asks user which element is being calibrated
			element = input("\nWhich element is being calibrated?: ")

			# Obtain binning
			binning = np.arange(len(data))

			# Calibrate binning
			energies = m*binning + b

			# Create a dictionary with the calibrated data
			df = {
				'energy (keV)': energies,
				'counts': data
			}

			# Convert dictionary to pandas dataframe
			df = pd.DataFrame.from_dict(df)

			# Ask user if they want to set x-bounds
			doBounds = input("Input x-bounds? (y/n): ")

			if doBounds == 'y':

				# Get bounds from user
				xStart = get_numerical_input("Input x-start: ")
				xEnd = get_numerical_input("Input x-end: ")

			else:
				xBounds = None
			
			# Check length (bins) of spectrum
			nBins = len(data)

			# Clipping noise (about bin 100 or 1/10 of the spectrum)
			cutoff = int(nBins/10)
			clip_val = np.max(data[cutoff:])*1.5

			# Clip the spectrum to remove noise
			data = np.clip(data, a_min=0, a_max=clip_val)

			##Step 3. Display the calibrated spectrum

			# Plot parameters
			plotArgs = {
				'color': 'b',
				'label': 'data',
				'xlabel': 'keV',
				'ylabel': 'counts',
				'title': f'{element} calibrated spectrum',
				'legend': True
				}

			# Plot and display the spectrum using plot_data function from plot.py
			plot.plot_data(energies, data, plotArgs, xBounds=xBounds)

			## Step 4: Save the calibrated spectrum as a CSV file and image

			# Save figure using create_image function in files.py
			files.create_image(element +'_calibrated_spectrum', folder_path)
			
			plt.show()

			# Save calibrated spectrum dataframe as CSV file using create_csv function in files.py
			files.create_csv(df, element +'_calibrated_spectrum', folder_path)

			print(f"\nCalibrated {element} spectrum.")
			
			total_items_processed += 1

		## If an error is raised, the program will continue to iterate through the list of files
		except Exception as e:

			# Tells user which file caused error and continues
			print(f"Error processing {data_processed.filename}")
			print("Please reprocess this file.")
			print("Skipping file....")

			# If there is a different error, tell user
			print(f"Error Message: {e}")	

		finally:
			
			# Print progress
			print(f"{total_items_processed} of {num_items_to_process} files processed.")

	return
