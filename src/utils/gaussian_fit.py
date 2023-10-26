import numpy as np
from scipy.optimize import curve_fit

### Defining a Gaussian distribution of total counts
### such that the integral over all space is 'A'
def gaussian(x, A, sigma, mu):
	"""
	Returns the value of a Gaussian function at a given point x, with specified amplitude, standard deviation, and mean.

	Parameters:
		x (array-like): The point at which to evaluate the Gaussian function.
		A (array-like): The amplitude of the Gaussian function.
		sigma (float): The standard deviation of the Gaussian function.
		mu (float): The mean of the Gaussian function.

	Returns:
		float: The value of the Gaussian function at the given point x.
	"""
	return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

### Defining a Gaussian distribution for two peaks
def double_gaussian(x, A1, sigma1, mu1, A2, sigma2, mu2):
	return gaussian(x, A1, sigma1, mu1) + gaussian(x, A2, sigma2, mu2)

def gaussian_fit(x, y, sigFigs=4):
	"""
	Fits a Gaussian function to the input data.

	Args:
		x (array-like): The x-values of the data.
		y (array-like): The y-values of the data.
		sigFigs (int, optional): The number of significant figures to round the output parameters to. Defaults to 4.

	Returns:
		tuple: A tuple containing the parameters of the fitted Gaussian function, the covariance matrix of the parameters, and the maximum y-value of the fitted function.
	"""
	## Find peak value of line
	max_counts = np.amax(y)

	## Find half the peak
	half_max = max_counts//2

	## Find all points above half the peak
	half_max_mask = (y >= half_max)

	## Find start and end X
	startX = x[half_max_mask][0]
	endX = x[half_max_mask][-1]

	## Estimate standard deviation with FWHM
	std_estimate = (endX - startX)/2.2

	## Find energy of spectral line
	peak_energy = x[np.where(y == max_counts)[0][0]]

	## Estimate for integral for area A
	A_estimate = 2.2 * max_counts * std_estimate

	print('Estimated fit parameters:')
	print('A = {:.7g}'.format(A_estimate))
	print('σ = {:.7g}'.format(std_estimate))
	print('μ = {:.7g}\n'.format(peak_energy))

	## Fit the gaussian to the ROI
	params_gauss, params_covariance = curve_fit(gaussian, x, y, p0=[A_estimate, std_estimate, peak_energy])

	## Make sure fit parameters are positive
	params_gauss = np.absolute(params_gauss)

	print('Computed fit parameters:')
	print(f'A = {A_estimate:.7g}')
	print('σ = {:.7g}'.format(params_gauss[1]))
	print('μ = {:.7g}\n'.format(params_gauss[2]))

	return params_gauss, params_covariance, max_counts


def integrate_gaussian(x, y, params_gauss, sigmas=3):
	"""
	Integrate a Gaussian function over a given range.

	Args:
		x (array-like): The x-values of the data.
		y (array-like): The y-values of the data.
		params_gauss (array-like): The parameters of the Gaussian function.
		sigmas (float, optional): The number of standard deviations to integrate over. Defaults to 3.

	Returns:
		float: The total counts of the integrated Gaussian function.
	"""
	# Find starting x-value (μ - sigmas*σ)
	startX = params_gauss[2] - sigmas*params_gauss[1]

	# Find ending x-value (μ + sigmas*σ)
	endX = params_gauss[2] - sigmas*params_gauss[1]

	# Get mask corresponding to this range of x-values
	mask = (x >= startX) & (x <= endX)

	# Slice y-values within this range
	y_slice = y[mask]

	# Sum counts
	total_counts = np.sum(y_slice)

	### Return total counts
	return total_counts