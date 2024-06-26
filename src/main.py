import argparse

import utils.files as files
import utils.spectrum as spectrum
import utils.calibrate as calibrate
import utils.plot as plot
import utils.simulated as sim

def main():
	"""
	This script provides a command-line interface for processing and displaying spectra.

	Usage:
	-cs, --calibrate: create a calibration curve from multiple points
	-cp, --cpoints: process a spectrum
	-d, --display: display a calibrated spectrum
	-r, --resolution: determine the resolution of the spectrometer
	-rp, --response: determine the response of the spectrometer
	"""
	parser = argparse.ArgumentParser(description="Process and display spectra.")

	parser.add_argument("-cc", "--calibration",
						help="creates a calibration curve from multiple points",
						action="store_true")
	parser.add_argument("-cp", "--cpoints", 
						help= "reads spectrum data and performs a Gaussian fit over a selected region **Multiple files must be from the same element.", 
						action="store_true")
	parser.add_argument("-cs", "--calibrate", 
						help="calibrate and display a spectrum", 
						action="store_true")
	parser.add_argument("-r", "--resolution", 
						help="determine the resolution of the spectrometer", 
						action="store_true")
	parser.add_argument("-rp", "--response", 
						help="determine the response of the spectrometer", 
						action="store_true")

	parser.add_argument("-ex", "--expected", 
						help="expected spectra", 
						action="store_true")

	args = parser.parse_args()

	if args.expected:
		sim.smooth_isotope_spectrum()

	if args.calibration:
		"""
		Creates a calibration curve from a set of input files containing calibration points.

		Step 1. Choose a folders or files containing calibration points using the function spectrum.process_spectrum()
		Step 2. Combine data as needed from all files
		Step 3. Create a calibration curve using the combined data and calling the function calibrate.create_calibration_curve()
		Step 4. Save the calibration curve to a file
		Step 5. Display the calibration curve and save image to a file using the function files.create_image()
		"""
		calibrate.create_calibration_curve()

	if args.cpoints:
		"""
		Select regions of interest in a spectrum and fit Gaussian curves to them.
		The results from the Gaussian fit are saved as a CSV file.

		Step 1. Choose a file(s) containing spectrum data using the function files.load_files()
		Step 2. Read the spectrum data from the file(s) using the function spectrum.read_spectra()
		Step 3. Plot the spectrum data and select region of interest using the function plot.select_peak()
		Step 4. Perform a Gaussian fit over the selected region using the function spectrum.gaussian_fit()
		Step 5. Save the Gaussian fit parameters to an array
		Step 6. Repeat steps 3-5 for each region of interest
		Step 7. Save the Gaussian fit parameters to a csv file using the function files.create_csv()
		"""
		calibrate.calibration_points()

	if args.calibrate:
		"""
		Calibrates spectral data using a calibration curve and saves the calibrated data as a CSV file and image.
		
		Step 1. Upload a spectrum file and calibration curve file using the function calibrate.calibrate_spectrum()
		Step 2. Calibrate the spectrum using the function calibrate.calibrate_spectrum()
		Step 3. Display the calibrated spectrum using the function plot.display_spectrum()
		Step 4. Save the calibrated spectrum to a csv file using the function files.create_csv()
		"""
		calibrate.calibrate_spectrum()

	if args.response:
		"""
		***Create this function: calibrate.determine_response()***

		Step 1. Load file of expected spectrum from STEAM's IDL code and Gaussian fit file using function calibrate.determine_response()
		Step 2. Pull maxCounts from Gaussian fit file
		Step 3. Plot maxCounts vs expected_counts using plot.plot_data()
		Step 4. Save the response to a csv file using the function files.create_csv()
		Step 5. Save the plot to an image file using the function files.create_image()
		
		"""
		calibrate.determine_response()

	if args.resolution:
		"""
		Determines the detector resolution by analyzing the Full Width Half Max (FWHM) and Energy (E) of the calibration points.
		Saves the detector response data to a CSV file and displays a plot of FWHM vs E.
		
		Step 1. Load a file(s) containing Gaussian fit parameters using the function files.load_files()
		Step 2. Determine the resolution of the spectrometer using the function calibrate.determine_resolution()
		Step 3. Plot sigmas vs energy using plot.plot_data()
		Step 4. Save the resolution to a csv file using the function files.create_csv()
		Step 5. Save the plot to an image file using the function files.create_image()
		"""
		calibrate.determine_resolution()

if __name__ == "__main__":
	main()

