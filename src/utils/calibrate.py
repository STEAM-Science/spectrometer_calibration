def calibration_points():

	"""

	It extracts the element name from the file name, reads the corresponding NIST data, and prompts the user to select peaks to fit.
	It then fits the selected peaks with Gaussian functions and calculates the calibration points.
	The calibration points are saved in a CSV file in the 'cPoints' directory.
	If the calibration curve already exists, the function exits without performing any operation.

	Args:
		filePath (str): The path of the file containing the spectrum data.
	"""
	print("Getting calibration points...")

	# ... do something with the spectral data ...
	return

def create_calibration_curve():
	"""
	Creates a calibration curve from a set of input files containing calibration points.

	The function prompts the user to input the file paths for each set of calibration points.
	It then calculates the calibration curve and saves the results to a CSV file.
	Additionally, it saves a plot of the calibration curve and displays it to the user.
	"""

	print("Creating calibration curve...")

	return