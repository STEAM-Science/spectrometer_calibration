# Import necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import periodictable as pt
from scipy.ndimage import gaussian_filter
from itertools import islice

import os

import utils.attenuation as atten

# Function to get user input
def get_input(prompt, default_value, input_value):

	try:
		# Get user input
		value = input(prompt)

		# If user input is empty, return default value
		if value == "":
			return default_value
		else:
			# Otherwise, convert user input to the desired type and return
			return input_value(value)

	## Handle error exceptions
	# If the user enters a value that cannot be converted to the desired type, print the error and use the default value
	except ValueError as ve:
		print(f"Invalid input: {ve}. Using default value.")
		return default_value

	# If any other error occurs, print the generic error and use the default value
	except Exception as e:
		print(f"An error has occured: {e}. Using default value.")
		return default_value

# Function to select detector and set default values
def select_detector():

	# Ask user to select the detector first using get_input function
	detector_select = get_input("Select detector (0 = SDD (default), 1 = CdTe) : ", 0, int)
	
	# Set default values based on detector selection
	if detector_select == 0:  # SDD
		defaults = {
			"resolution": 0.15,  # keV FWHM
			"aperture": 300,  # microns
			"wid_ee": (20 - 0.5) / 1024
		}

	elif detector_select == 1:  # CdTe
		defaults = {
			"resolution": 0.3,  # keV FWHM
			"aperture": 2700,  # microns
			"wid_ee": (100 - 1.) / 1024
		}

	# Return detector_select and default dictionary
	return detector_select, defaults

# Function to calculate bin edges, centers, and widths
def get_edges(array, num_bins):

    # Calculate the bin edges
    bin_edges = np.histogram_bin_edges(array, bins=num_bins)

    # Calculate the bin widths
    bin_widths = np.diff(bin_edges)

    # Calculate the bin centers (approximation for mean energy)
    bin_centers = bin_edges[:-1] + bin_widths / 2

	# Return the bin edges, centers, and widths
    return bin_edges, bin_centers, bin_widths

# Main function to simulate a spectrum
def smooth_isotope_spectrum():
	
	try:
		# Ask user to select the detector first using the select_detector function
		detector_select, defaults = select_detector()

		# Set default values based on detector selection
		resolution = defaults["resolution"]
		aperture = defaults["aperture"]

		# Isotope dictionary
		isotopes = {
			"Am": {"activity": 1, "half_life": 432.2}, #activity in micro Curies, half-life in years
			"Ba": {"activity": 10, "half_life": 10.5},
			"Cd": {"activity": 10, "half_life": 1.27},
			"Fe": {"activity": 100, "half_life": 2.7},
			"Zn": {"activity": 10, "half_life": 0.668},
		}

		# Print instructions for user MOVE
		print("\nFollow the prompts below. Press enter to use the default value.")

		# User input, if needed add "options" to the inputs dictionary
		inputs = {
			"detector_select": {"prompt": "Select detector (0 = SDD (default), 1 = CdTe) : ", "default": 'detector_select', "type": int},
			"resolution": {"prompt": f"Enter detector resolution in keV FWHM (default is {resolution}): ", "default": resolution, "type": float},
			"aperture": {"prompt": f"Enter aperture radius in microns (default is {aperture}): ", "default": aperture, "type": int},
			
			"filter": {"prompt": "Enter filter material (default is Be): ", "default": "Be", "type": str},
			"filter_thick": {"prompt": "Enter filter thickness in microns (default is 0): ", "default": 0, "type": float},

			"element": {"prompt": "Enter isotope without isotope identification (ex: Fe-55 -> Fe) (default is Fe-55): ", "default": "Fe", "type": str},
			"activity": {"prompt": "Enter isotope activity in microCuries (default is 100): ", "default": 100, "type": float},

			"distance": {"prompt": "Enter isotope distance from detector in cm (default is 1): ", "default": 1., "type": float},
			"time": {"prompt": "Enter isotope integration time in seconds (default is 1): ", "default": 100., "type": float},
			"days": {"prompt": "Enter days since isotope purchase (default is 1): ", "default": 1, "type": int},

			"wid_ee": {"prompt": "Complete. Press enter to continue:", "default": defaults["wid_ee"], "type": float},
				}

		# Append the user input from prompts to inputs dictionary
		total_questions = len(inputs)-2
		for i, (key, value) in enumerate(islice(inputs.items(), 1, None), start=1):
			if i == total_questions+1:
				prompt = value["prompt"]
			else:
				prompt = f"({i}/{total_questions}): {value['prompt']}"
			
			inputs[key]["value"] = get_input(prompt, value["default"], value["type"])

			if key == "isotope" and key == "filter":
				lower_case = inputs[key]["value"].lower()
				inputs[key]["value"] = lower_case.capitalize()

		print("\nUser inputs stored...")
		print("Continuing with expected measurements...")

		# Set variables from inputs dictionary
		filter_material = inputs["filter"]["value"]
		filter_thick = inputs["filter_thick"]["value"]

		element = inputs["element"]["value"]
		activity = inputs["activity"]["value"]
		distance = inputs["distance"]["value"]
		time = inputs["time"]["value"]
		days = inputs["days"]["value"]

		wid_ee = defaults["wid_ee"]

		# Exoponentially decay of the activity
		# Calculate the decay constant, convert half-life from years to days
		decay_constant = np.log(2) / (isotopes[element]["half_life"] * 365)

		# Calculate the activity after a certain number of days
		activity_after_decay = activity * np.exp(-decay_constant * days)

		print(f"\nActivity of {element} after {days} days: {activity_after_decay:.4g} microCuries")

		# Calculate area in cm^2
		area = np.pi*(aperture*1e-4/2.)**2 
			
		# Create arbitrary energy array from 0.5 to 20 keV with 0.01 keV steps (bins)
		wid_ee = (20 - 0.5)/1024

		# Create arbitrary energy array, 0.5-20 keV, ~0.02 keV bins
		energy_array = np.arange((20.-0.5)/wid_ee+1)*wid_ee+0.5

		# Call to get_edges function
		eee, eee_mean, eee_wid = get_edges(energy_array, num_bins=10239)

		# Make new edges for fine energy array
		wid_ee2 = (20 - 0.5)/1024
		energy_array2 = np.arange((20.-0.5)/wid_ee2+1)*wid_ee2+0.5

		# Call to get_edges function
		eee2, eee_mean2, eee_wid2 = get_edges(energy_array2, num_bins=1023)


		# Create spectrum
		# Idl pro files: make_isotope_spectra, instrument_response, air_attenuation 
		

		# make_isotope_spectra.pro
		print(f"\nCalculating {element} spectrum...")
		spectrum = area * make_isotope_spectra(eee, element, activity_after_decay, time) / (4.*np.pi*distance**2)
		
		# instrument_response.pro (simulated, NOT from calibrate.py)
		print("Calculating instrument response...")
		#response = instrument_response(eee_mean2, detector_select, filter_thick, filter_material)
		response = [0]
		resp_atten = instrument_response(eee_mean2, detector_select, filter_thick, filter_material)
		response = np.append(response, resp_atten)

		# air_attenuation.pro
		print("Calculating air attenuation...")
		air_attenuation = [0]
		air_atten = atten.air_attenuation(eee_mean2, distance)
		air_attenuation = np.append(air_attenuation, air_atten)

		# now need gaussfold (line 84 of IDL) its a built in idl thing cries
		# needs to be spectrum, sigma <_ reagrranged FWHM equation


		print("Smoothing spectrum using gaussian filter...")
		print(f'\nresolution: {resolution}, eee_mean: {eee_mean2}')
		sigma = np.mean(resolution/2.355/eee_wid)
		print(f'sigma: {sigma}')
		print(spectrum)
		smooth_spectrum = gaussian_filter(spectrum, sigma, mode='constant', cval=0.0)
		smooth_spectrum = np.append(smooth_spectrum, 0)
		

		# Rebin the spectrum
		#from bin 0 to 9, that is bin 1 and so on
		smooth_spectrum_rebinned = []
		for i in range(0, len(smooth_spectrum), 10):
			rags = smooth_spectrum[i:i+10] #current chunk

			#sum the chunk
			rags_sum = sum(rags)
			smooth_spectrum_rebinned.append(rags_sum)
		
		print(len(smooth_spectrum_rebinned))

		for x in response:
			if x != 0:
				raise ValueError("Response is zero, damn")

		smooth_spectrum_rebinned = smooth_spectrum_rebinned*response
		# Add Poisson noise
		smooth_spectrum_rebinned_noise = atten.add_poisson_noise(smooth_spectrum_rebinned)
		#print("\nSpectrum rebinned with Poisson noise: ", spectrum_rebinned)
		# print("\nspectrum rebinned: ", spectrum_rebinned)
		# print("\neee2: ", eee2)

		print("\nPlotting spectrum...")
		print("Displaying spectrum.")

		# Plot the rebinned spectrum
		print(f'Length of eee2: {len(eee2)} and Length of smooth_spectrum_rebinned_noise: {len(smooth_spectrum_rebinned_noise)}')
  
		plt.plot(eee2, smooth_spectrum_rebinned_noise)
		plt.legend()
		#plt.plot(eee_mean2, spectrum_rebinned, label='spectrum')
		plt.title(f'{element} Expected Spectrum')
		plt.xlabel('Energy (keV)')
		plt.ylabel('Counts (photons)')
		plt.show()

		# Creating the output_spectrum dictionary
		output_spectrum = {'element': element, 'energy': eee2, 'counts': smooth_spectrum_rebinned_noise}
		
		return output_spectrum
	
	except ValueError as e:
		print(f"\nAn error has occured: {e}.")
		return
	except Exception as e:
		print(f"\nAn error has occured: {e}.")
		return
	

def make_isotope_spectra(energy_edges, element, activity, time):

	# This is me checking to make sure that whatever tf is going on in the function above works 
	# Check to make sure energy edges is not empty
	if len(energy_edges) == 0:
		print("Missing energy edge input...Returning -1")
		return -1
	
	# Defining constants
	decays_per_second = 3.7e10 # [micro curie]

	# Get keV and intensity from NIST data csv file
	# Get root directory
	root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	

	# Construct path to NIST data csv file		
	nist_path = f'{root_dir}/nist_data/Spectral_Lines_{element}.csv'

	# Load csv file into pandas dataframe
	df = pd.read_csv(nist_path, skiprows=1, names=["keV", "intensity"], usecols=[0, 1])

	# Convert dataframe columns to numpy arrays
	energy = df["keV"].to_numpy()
	intensity = df["intensity"].to_numpy()

	# Calculate number of photons (Count Rate), in photons per second
	number_of_photons = intensity * activity * decays_per_second * time

	# Makes a zero array with length of energy edges array -1
	output_spectrum = np.zeros(len(energy_edges) -1)

	for i in range(len(energy)):
		index = np.min(np.where(energy_edges > energy[i]))

		if index > 0:
			output_spectrum[index] += number_of_photons[i]

	return output_spectrum
	
def instrument_response(energy_centers, detector_select=0, filter_thick=None, filter_material='Be'):
	# Define constants
	a2kev = 12.398420
	kev_per_el = 3.64e-3  # energy per electron/hole pair in Si
	si_thick = 500.  # microns
	cdte_thick = 1000.  # cm

	# Get XCOM attenuation factors
	root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	mu_be = np.loadtxt(f'{root_dir}/henke_model/be_1_100keV.dat', skiprows=2)
	mu_cdte = np.loadtxt(f'{root_dir}/henke_model/cdte_1_100keV.dat', skiprows=2)
	mu_energies = mu_be[0, :] * 1000  # keV
	polytran = np.loadtxt(f'{root_dir}/henke_model/polyimide_1_100Ang.dat', skiprows=2)  # wavelength in nm!!
	be_dens = 1.85  # g/cm^3
	cdte_dens = 5.85  # g/cm^3

	if filter_material == 'Al':
		mu_filt = 1
		#mu_filt = np.loadtxt(f'{root_dir}/henke_model/al_1_100keV.dat', skiprows=2)
		filt_dens = 2.70  # g/cm^3
	elif filter_material == 'Pl':
		mu_filt = np.loadtxt(f'{root_dir}/henke_model/polyimide_1_100Ang.dat', skiprows=2)
		filt_dens = 1.42  # g/cm^3
	else:
		mu_filt = mu_be
		filt_dens = be_dens

	# SDD
	if detector_select == 0:  
		be_thick = 15.  # micron
		filt_thick = filter_thick if filter_thick is not None else 0.  # micron
		resolution = 0.15  # keV FWHM

		if filter_material == 'Al':
			wv_besi, resp_besi = atten.diode_param(['Be', 'Al'], [be_thick*1e4, filt_thick*1e4], si_thick=si_thick*1e4, oxide_thick=70.)
			resp_poly = 1.
		elif filter_material == 'Pl':
			wv_besi, resp_besi = atten.diode_param(['Be'], [be_thick*1e4], si_thick=si_thick*1e4, oxide_thick=70.)
			resp_poly = np.interp(wv_besi, polytran[0, :]*10, polytran[1, :]**filt_thick)
		else:
			wv_besi, resp_besi = atten.diode_param(['Be'], [(be_thick+filt_thick)*1e4], si_thick=si_thick*1e4, oxide_thick=70.)
			resp_poly = 1.

		# Convert to cts/ph = (el/ph) / (keV / (keV / el))
		resp_besi /= (a2kev/wv_besi)/kev_per_el

		# Interpolate response (cts/ph) at eee bins, zero the response below 0.5 keV (LLD)
		resp = np.interp(energy_centers, a2kev/wv_besi, resp_besi*resp_poly)
		resp[energy_centers < 0.5] = 0.

		return resp

	# CdTe
	elif detector_select == 1:  
		be_thick = 100. * 1e-4  # cm
		filt_thick = filter_thick if filter_thick is not None else 0.  # cm
		resolution = 0.3  # keV FWHM

		# Calculate response of detector, cts/photon, added air??
		resp_raw = np.exp(-mu_be[3, :] * be_thick * be_dens) * np.exp(-mu_filt[3, :] * filt_thick * filt_dens) * (1 - np.exp(-mu_cdte[3, :] * cdte_thick * cdte_dens))

		# Interpolate response (cts/ph) at eee bins, zero the response below 5 keV (LLD)
		resp = np.interp(energy_centers, mu_energies, resp_raw)
		resp[(energy_centers < 4.5) | (energy_centers >= 100)] = 0.

		return resp
