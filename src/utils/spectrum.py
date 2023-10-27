import os
import re
import pathlib
import numpy as np
import pandas as pd

import utils.files as files
import utils.classes as classes

def process_spectrum():
	"""
	Prompts the user to choose between processing a folder or individual files.
	Loads the selected files and passes them to the read_spectrum function in the spectrum module.
	
	Returns:
		spectral_data (list): A list of spectral data obtained from the loaded files.
	"""
	user_input = input("Do you want to process a folder (f) or individual files (i)?: ")
	
	# Removing case sensitivity of user input
	user_input_case = user_input.lower()
	
	if user_input_case == "f":
		filenames = files.load_folder()
	elif user_input_case == "i":
		filenames = files.load_files()
	else: 
		raise ValueError("Invalid input")

	print(len(filenames), "files loaded")
	
	# If multiple files are loaded, checking with user if they want to sum the spectra
	if len(filenames) != 1:

		# Ask user if they want to sum the spectra
		user_input = input("Do you want to sum the loaded spectra? (y/n): ")
		user_input_case = user_input.lower()

		if user_input_case == "y":
			return sum_spectra(filenames)
		elif user_input_case == "n":
			return read_spectra(filenames)
		## TODO: if i want to further automate the process, i can check the filename for 
		# instances where element = element and THEN ask if the user wishes to sum
		# for filename in filenames:
		# element = element then ask

	if len(filenames) == 1:
		return read_spectra(filenames)

	print("\nDone reading all spectra!")
 


### Read uploaded spectrum file(s)
def read_spectra(filenames):
	"""
	Reads spectra from a list of files.

	Args:
		filenames (list): A list of filenames to read spectra from.

	Returns:
		spectral_data (numpy array): A list of spectral data.
	"""
	spectral_data = []

	## Obtaining data from loaded files	
	for filename in filenames:
		print("Reading spectrum from", filename.name)

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
				data = df.iloc[-1].to_numpy()

				print("Done reading spectrum from", filename.name)
				print(type(data))
				## Return data as array
				data_processed = classes.DataProcessor(filename.stem, data)
				append_data = spectral_data.append(data_processed)
		
		elif ext == ".txt":
			with open(filename) as file:

				# Lists to store lines from txt
				data = []

				lines = file.read().splitlines()

				# Iterate through all lines
				for line in lines:

					# Remove all non-data lines
					if not re.search('[a-zA-Z]', line):
				
						data.append(line)

				# Convert to array
				data = np.asarray(data).astype(int)
			
				# Append to spectral_data
				data_processed = classes.DataProcessor(filename.stem, data)
				spectral_data.append(data_processed)

				print("Done reading spectrum from", filename.name)
		
		else:
			raise ValueError("File format not supported! Please load a .csv or .txt file.")
	
	### Return data as an array
	return spectral_data

### Sum multiple spectra from the same element
def sum_spectra(filenames):
	"""
	Sums the spectra in the loaded list of filenames.

	Args:
		filenames (list): A list of filenames containing spectra data.

	Returns:
		spectral_data (numpy array): A list of spectral data.
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


