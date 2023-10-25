def process_spectrum():
	"""
	This function prompts the user to choose between processing a folder or individual files.
	It then loads the selected files and passes them to the read_spectrum function in the spectrum module.
	"""
	user_input = input("Do you want to process a folder (f) or individual files (i)? ")
	if user_input == "f":
		filenames = files.load_folder()
	elif user_input == "i":
		filenames = files.load_files()
	else: 
		raise ValueError("Invalid input")

	read_spectra(filenames)

def read_spectra(filenames):
	"""Reads spectra from a list of filenames.

	Args:
		filenames (list): A list of filenames to read spectra from.

	Returns:
		spectral_data (numpy array): A list of spectral data.
	"""
	for filename in filenames:
		print("Reading spectrum from", filename)

		# Check if csv or txt file

		print("Done reading spectrum from", filename)
	print("Done reading all spectra!")

def sum_spectra(spectral_data):
	# ... do something with the spectral data ...
	return spectral_data


