import os
import re
import pathlib
import numpy as np
import pandas as pd

import utils.files as files

def process_spectrum():
	"""
	This function prompts the user to choose between processing a folder or individual files.
	It then loads the selected files and passes them to the read_spectrum function in the spectrum module.
	"""
	user_input = input("Do you want to process a folder (f) or individual files (i)?: ")
	if user_input == "f":
		filenames = files.load_folder()
	elif user_input == "i":
		filenames = files.load_files()
	else: 
		raise ValueError("Invalid input")
		# If multiple files are loaded, checking with user if they want to sum the spectra
	
	read_spectra(filenames)

	if len(filenames) != 1:

		# Ask user if they want to sum the spectra
		user_input = input("Do you want to sum the loaded spectra? (y/n): ")
		user_input_case = user_input.lower()

		if user_input_case == "y":
			sum_spectra(filenames)
		else:
			pass
		## TODO: if i want to further automate the process, i can check the filename for 
		# instances where element = element and THEN ask if the user wishes to sum
		# for filename in filenames:
		# element = element then ask

### Read uploaded spectrum file(s)
def read_spectra(filenames):
	"""
	Reads spectra from a list of files.

	Args:
		filenames (list): A list of filenames to read spectra from.

	Returns:
		spectral_data (numpy array): A list of spectral data.
	"""

	## Obtaining data from loaded files	
	for filename in filenames:
		print("Reading spectrum from", filename)

		## Check if file exists
		if not os.path.isfile(filename):
			raise FileNotFoundError("File does not exist")

		## Get file format
		_, ext = os.path.splitext(filename)

		## Check if csv or txt file and read file
		if ext == ".csv":
			with open(filename) as file:

				# Read file
				df = pd.read_csv(file)

				# Get data (last row of .csv file) and convert to array
				spectral_data = df.iloc[-1].to_numpy()

				print("Done reading spectrum from", filename)
				## Return data as array
				return spectral_data

		elif ext == ".txt":
			with open(filename) as file:

				# Lists to store lines from txt
				spectral_data = []

				lines = file.read().splitlines()

				# Iterate through all lines
				for line in lines:

					# Remove all non-data lines
					if not re.search('[a-zA-Z]', line):
				
						spectral_data.append(line_list)
			
			## Return data as array
			return np.asarray(spectral_data).astype(int)

		else:
			raise ValueError("File format not supported! Please load a .csv or .txt file.")

		#print("Done reading spectrum from", filename)
	print("Done reading all spectra!")


### Sum multiple spectra from the same element
import numpy as np

def sum_spectra(filenames):
	"""
	Sums the spectra from a list of filenames.

	Args:
		filenames (list): A list of filenames containing spectra data.

	Returns:
		numpy.ndarray: An array containing the summed spectra data.
	"""
	print("Summing spectra....")

	## Initiating the array
	stored_data = read_spectra(filenames)

	## Empty array full of zeros to store summed spectra
	spectral_data = np.zeros_like(stored_data[0])

	## Sum all spectra
	for i in stored_data:
		spectral_data += i
	
	print("Summing complete!")

	### Return summed spectra as an array
	return spectral_data


