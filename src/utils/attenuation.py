import numpy as np
import periodictable as pt
import os
import struct
import pandas as pd

compounds_info = {
	'Polyimide': { # C22H10N2O5
		'compound': ['C', 'H', 'N', 'O'],
		'quantity': [22, 10, 2, 5],
		'atwt': 201,
		'density': 1.51,
		'atomic_number': ''},
	'SiO': {
		'compound': ['Si', 'O'],
		'quantity': [1, 1],
		'atwt': 44.09,
		'density': 2.65,
		'atomic_number': ''},
	}

def get_element_info(element): # this should get rid of compound and zcompound
	
	# Get the compound object from the periodictable module
	try:
		
		compound = getattr(pt, element)
		# Get the atomic weight
		atwt = compound.mass  # [amu]
		# Get the density
		density = compound.density  # [g/cm^3]
		# Get the atomic number
		atomic_number = compound.number 
		
		return atwt, density, atomic_number

	# If the element is not found in the periodic table, try to get it from the dictionary
	except AttributeError:
		
		if element in compounds_info:

			if element == 'Pl':
				compounds = compounds_info['Polyimide']['compound']
				quantity = compounds_info['Polyimide']['quantity']
				atomic_number_arrray = []

				# get atomic number of each element in compound
				for i in compounds:
		
					compound = getattr(pt, i)
					atomic_number_arrray.append(compound.number)

				# sum atomic numbers based on quantity
				sum_atomic_number = 0
				for i in range(len(quantity)):
					sum_atomic_number += quantity[i]*atomic_number_arrray[i]

				# 22 C, C-6 so f1 is 22*6 / sum
				frac_electrons = []
				for i in range(len(quantity)):
					compound = getattr(pt, compounds[i])

					frac_electrons = (quantity[i]*compound.number)/sum_atomic_number
					frac_electrons.append(frac_electrons)

				# do wiki forumla with each atomic number
				effective_atomic_number = 0
				for i in range(len(frac_electrons)):
					effective_atomic_number += (
						frac_electrons[i] * atomic_number_arrray[i] ** 2.94
					)
					
				return atwt, density, effective_atomic_number**(1/2.94)
			
			# TODO: ADD else if for other compounds (SiO, etc.)
   
			else: 
				atwt = compounds_info[element]['atwt']  # [amu]
				# Get the density
				density = compounds_info[element]['density']  # [g/cm^3]
				# Get the atomic number
				atomic_number = compounds_info[element]['atomic_number']

				return atwt, density, atomic_number
		else:
			# If the element is not in the dictionary, return an error message
			raise ValueError(f"The element '{element}' does not exist in the periodictable library or in the compounds_info dictionary.")

    
	# Return the atomic weight and density
	

def henke_array(element, density, graze_mrad=0):

    # Your code for handling different system files goes here

	try:
		atwt, density, atomic_number = get_element_info(element)

		root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	
		# Find henke.dat		
		henke_data = f'{root_dir}/henke_model/henke.dat'
		
		print(f"STOOOOOOOOOOOOOOOOOP\n")
		
		with open(henke_data, 'rb') as file:
			# Read the number and elements from the file
			num_energies, num_elements = struct.unpack('ii', file.read(8))
			energies = np.fromfile(file, dtype=np.float32, count=num_energies)
			f1 = f2 = this_f1 = this_f2 = np.zeros(num_energies)

			# header_offset = 4 * (2 + num_energies)
			# maxz = len(atomic_number)
			# for i in range(maxz):
			# 	if atomic_number[i] > 0.:
			# 		offset = header_offset + i * 2 * num_energies * 4
			# 		file.seek(offset)
			# 		this_f1 = np.fromfile(file, dtype=np.float32, count=num_energies)
			# 		this_f2 = np.fromfile(file, dtype=np.float32, count=num_energies)
			# 		this_f1 = this_f1.byteswap() if this_f1.dtype.byteorder == '<' else this_f1
			# 		this_f2 = this_f2.byteswap() if this_f2.dtype.byteorder == '<' else this_f2
			# 		f1 = f1 + z_array[i] * this_f1
			# 		f2 = f2 + z_array[i] * this_f2

			# Defining constants
			AVOGADRO = 6.02204531e23  # Avogadro's number
			HC = 12398.54  # Planck's constant times speed of light [eV*Angstrom]
			RE = 2.817938070e-13  # Electron radius [cm]

			# Calculate the number of moledules per cubic centimeter (if statment to avoid zero error)
			if atwt != 0.0:
				molecules_per_cc = density * AVOGADRO / atwt
			else:
				molecules_per_cc = 0.0

			wavelength = HC / energies

			# This constant has wavelength in Angstroms, and then they are converted to cm
			constant = RE * (1.0e-16 * wavelength * wavelength) * molecules_per_cc / (2.0 * np.pi)

			delta = constant * f1[:len(constant)]
			beta = constant * f2[:len(constant)]

			if graze_mrad == 0:  # Assuming normal incidence
				reflect = 1. + np.zeros(num_energies)
			else:
				# Calculate the reflectivity for a non-zero graze angle
				#reflect = calculate_reflectivity(graze_mrad, delta, beta)
				pass

	except FileNotFoundError:
		print(f'Could not open file "henke.dat" or {henke_data}')
		raise

	return f1, f2, energies, delta, beta, reflect


def henke_t(element_name, density):

	# Print message to track progress	
	print('henke_t,element_name,wavelength,mu')
	print('    where mu = 1/e absorption length in Angstroms')

	# Get compound properties from get_element_info function
	atwt, density, atomic_number = get_element_info(element_name)
	print(f'Information for {element_name}: {atwt:.4f} amu, {density:.4f} g/cm^3, {atomic_number}')

	# Call the henke_array function to calculate f1, f2, energies, delta, beta, reflect
	f1, f2, energies, delta, beta, reflect = henke_array(element_name, density)

	# Calculate the absorption 1/e and wavelength
	mu = (1.238 / (energies*4*np.pi*beta) ) * 1e4  # convert to Angstoms
	wavelength = 12397 / energies  # convert to wavelength (Angstroms)
	

	return wavelength, mu

# This function is a direct translation of henke_t.pro from IDL to Python (See STEAM's IDL Henke module folder)
def diode_param(material, thickness, si_thick=50000, oxide_thick=70):
	"""
	Note: This function is a direct translation of henke_t.pro from IDL to Python (See STEAM's IDL Henke module folder)
	- current1 is calculated using a simple formula that assumes each photon with energy of 3.65 eV produces one electron. 
	This is a common approximation used in photodiode calculations.

	- current2 is calculated using a more complex method that takes into account the actual quantum efficiency (s_qe) of the diode at different wavelengths. 
	This is typically more accurate than the simple approximation used to calculate current1.

	The line 315 current = current1 sets current to the value of current1. 
	This means that the function is effectively using the simple approximation to calculate the sensitivity, regardless of the more complex calculation stored in current2.
	
	Comment out line 315 and uncomment line 316 to use the more complex calculation instead.

	""" 

# Check if the number of materials matches the number of thicknesses
	try:
		assert len(material) == len(thickness)

		# Set the minimum thickness for silicon and oxide
		si_thick = max(si_thick, 1000)
		oxide_thick = max(oxide_thick, 10)

		# Set the oxide material
		oxide_material = 'SiO'

		# Initialize variables
		ename = ''
		tname = ''
		num = 0
		el = ''
		thick = 0.0

		# Loop over all materials
		for k in range(len(material)):
			# Get the current material and thickness
			el = material[k].strip()
			thick = thickness[k]

			# Call the henke_t function to calculate wavelength and mu
			wv, mu = henke_t(el, thick)

			# Update the element name and thickness strings
			ename += '/' + el if num > 0 else el
			tname += '/' + str(int(thick)) if num > 0 else str(int(thick))

			# Calculate the transmission
			if num == 0:
				trans = np.ones_like(wv)
			trans *= np.exp(-1.0 * thick / mu)

			num += 1

		# Get the oxide transmission
		wv, mu = henke_t(oxide_material, oxide_thick)
		trans *= np.exp(-1.0 * oxide_thick / mu)

		# Get the silicon absorption
		wv, mu = henke_t('Si', si_thick)
		si_abs = 1.0 - np.exp(-1.0 * si_thick / mu)

		# Calculate the number of electrons per each 3.65 eV of photon energy
		factor = 6.624E-34 * 2.998E8 / 3.64 / 1E-10 / 1.602177E-19
		current1 = trans * si_abs * factor / wv

		w_qe = [0.10,  0.13,  0.17,  0.22,  0.28,  0.36,  0.46,  0.60, 
			0.77,  1.00,  1.29,  1.66,  2.14,  2.77,  3.57,  4.61, 
			5.95,  7.69,  9.92, 12.81, 16.54, 
			17, 22, 27, 32, 37, 42, 47, 
			50.,   53,   58,   63,   68,   73,   78,   83,   90,   95,  100, 
			106.,  116,  126,  136,  146,  156,  166,  175,  185,  195,  205,
			215.,  225,  235,  245,  255,  270,  290,  310,  330,  350,  370, 
			390.,  410,  430,  450,  470,  490,  519,  537,  556,  584,  599, 
			622.,  639,  657,  669,  683,  699,  712,  735,  752,  771,  800, 
			818,  844,  865,  886,  920,  1026, 1164, 1180, 1216, 1254, 1354, 
			1403, 1441, 1487, 1545, 1608, 1648, 1700, 1750, 1823, 1879, 1937, 
			2000, 2067, 2138, 2214, 2296, 2385, 2537 ]

		# Define the sensitivity values
		s_qe = [30575.340, 23683.605, 18345.275, 14210.218, 11007.208,  8526.164,  6604.351,  5115.719, 
			3962.626,  3069.443,  2377.585,  1841.673,  1426.557,  1105.008,   855.937,   663.008, 
			513.564,   397.806,   308.140,   238.685,   184.885, 
			179.8, 138.9, 113.2, 95.52, 82.61, 72.78, 65.03, 
			62.46,58.36,52.48,49.80,42.21,35.32,33.09,32.46,31.22,28.85,27.53, 
			25.94,23.34,22.72,21.45,19.89,17.85,16.88,15.43,14.55,13.55,12.78, 
			12.45,11.80,11.25,10.66,10.08, 9.60, 8.80, 8.38, 7.91, 7.34, 7.01, 
			6.68, 6.32, 5.92, 5.58, 5.33, 5.08, 4.70, 4.37, 4.05, 3.62, 3.52, 
			3.25, 3.01, 2.78, 2.64, 2.48, 2.32, 2.20, 2.01, 1.92, 1.80, 1.68, 
			1.57, 1.47, 1.40, 1.33, 1.27, 1.02, 1.06, 1.05, 1.12, 1.12, 1.10, 
			1.08, 1.07, 1.03, 1.02, 0.96, 0.88, 0.79, 0.74, 0.69, 0.65, 0.62, 
			0.57, 0.54, 0.51, 0.48, 0.47, 0.47, 0.39 ]

		current2 = trans * si_abs * np.interp(wv, w_qe, s_qe)

		#current = current2
		current = current1

		return wv, current

	except AssertionError:
		print('ERROR: material and thickness must be same size array!')
		raise	
	except Exception as e:
		print(f'ERROR: {e}')
		raise

def air_attenuation(energy_centers, distance=1, air_dens=None):
	a2kev = 12.398420
	kev_per_el = 3.64e-3  # energy per electron/hole pair in Si
	air_thick = distance  # in cm

	# Get XCOM attenuation factors
	root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	
			
	mu_air = np.loadtxt(f'{root_dir}/henke_model/air_1_100keV.dat', skiprows=2)
	mu_energies = mu_air[0, :] * 1000  # keV

	if air_dens is None:
		air_dens = 0.001049  # change to being an input pressure and calculate density from it

	# Calculate air attenuation
	resp_raw = np.exp(-mu_air[3, :] * air_thick * air_dens)

	# Interpolate response (cts/ph) at eee bins, zero the response below 5 keV (LLD)
	resp = np.interp(energy_centers, mu_energies, resp_raw)
	resp[(energy_centers < 0.5) | (energy_centers >= 100)] = 0

	return resp

def add_poisson_noise(spectrum):
	"""
	Adds Poisson noise to data.
	"""
	# Initialize the noisy spectrum
	noisy_spectrum = np.zeros_like(spectrum)

	SCALE_FACTOR = 1e6  # Adjust this value as needed

# Loop over each element in the spectrum
	for i in range(len(spectrum)):
		# If the spectrum value is greater than 0, add Poisson noise
		if spectrum[i] > 0:
			# Scale down the spectrum value
			scaled_spectrum = spectrum[i] / SCALE_FACTOR
			# Add Poisson noise to the scaled value
			noisy_spectrum[i] = np.random.poisson(scaled_spectrum)
			# Scale the noisy spectrum value back up
			noisy_spectrum[i] *= SCALE_FACTOR
		# Otherwise, set the noisy spectrum value to 0
		else:
			noisy_spectrum[i] = 0
			# Otherwise, set the noisy spectrum value to 0

	return noisy_spectrum
	