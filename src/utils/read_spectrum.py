### Read uploaded spectrum file and display it

### Importing useful packages
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.optimize import curve_fit
import glob
import cv2
import argparse
from scipy.stats import norm
import datetime
from mpl_point_clicker import clicker

### Importing useful scripts
from gaussian_fit import *
from llsfit import *

### Read uploaded spectrum file and display it
def read_spectrum(filepath):

    ## Get file format
    _, ext = os.path.splitext(filepath)

    ## Ask user for file format
    user_file = input('Is the file a .csv or .txt?: ')

    # Convert user input to lowercase
    user_file_case = user_file.lower()

    ## If file is a .csv continue to the next step
    if user_file_case == 'csv':

        print(f'Reading file: {filepath}')

        # Check file format and continue
        if ext == '.csv':

            # Go to .csv function read
            csv_read(filepath)
      
        else:
            raise ValueError('File format not .csv!')

    ## If file is a .txt continue to the next step
    elif user_file_case == 'txt':

        print(f'Reading file: {filepath}')

        # Check file format and continue
        if ext == '.txt':

            # Go to .txt function read
            txt_read(filepath)

        else:
            raise ValueError('File format not .txt!')

    else:
            
        raise ValueError('File format not supported!')

### Read .csv file and save spectrum data
def csv_read(filepath):

    ## Read file
    with open(filepath) as file:
        df = pd.read_csv(filepath)

        ## Get data (last row of .csv file) and convert to array
        spectral_data = df.iloc[-1].to_numpy()

    ### Return data
    return spectral_data

### Read .txt file and save spectrum data
def txt_read(filepath):

    ## Read file
    with open(filepath) as file:

        # Lists to store lines from txt
        spectral_data = []

        lines = file.read().splitlines()
    
        # Iterate through all lines
        for line in lines:

            # Remove all lines that do not contain numbers
            if not re.search('[a-zA-Z]', line):
            
                spectral_data.append(line)
    
    # Return data as an array
    return np.asarray(spectral_data).astype(int)

### Plot spectrum data
def plot_data(x, y, plotArgs, xBounds=None, yBounds=None):

    # Clear existing plots
    plt.clf()

    # Plot whole spectrum
    plt.plotData(x, y, color=plotArgs['color'], label=plotArgs['label'])

    # If user specified x-bounds
    if xBounds != None:

        # Set x-bounds
        plt.xlim(xBounds)
    
    # If user specified y-bounds
    if yBounds != None:

        # Set y-bounds
        plt.ylim(yBounds)
    
    # Add title
    plt.title(plotArgs['title'])

	# Add axes labels
    plt.xlabel(plotArgs['xlabel'])
    plt.ylabel(plotArgs['ylabel'])

	# If user wants to show a legend
    if plotArgs['legend']:

		# Add legend
        plt.legend()

### Display spectrum and allow user to select peaks
def select_peak(bins, spectrum, element, clipVal=0, doublet=False):

    ## TODO: Cannont find instance where the user is asked where to clip
    ## the data, either add it or delete it --> I deleted it

    # Check how many bins the spectrum contains
    nBins = len(spectrum)

    # Clipping noise (about bin 100 or 1/10 of the spectrum)
    cutoff = int(nBins/10)

    # Find the largest value in the spectrum after cuffoff
    clipVal = np.max(spectrum[cutoff:])*1.5

    # Clip spectrum at this point and remove noise
    spectrum = np.clip(spectrum, a_min = 0, a_max = clipVal)

    ## Plot and display the spectrum
    plotArgs = {
        'color': 'k',
		'label': 'data',
		'xlabel': 'bin',
		'ylabel': 'counts',
		'title': f'{element}: Zoom into peak',
		'legend': True
    }

    ## Zoom into the region of interest
    # Plot spectrum
    plot_data(bins, spectrum, plotArgs)

    # Get input points from user (click 2 points on the display)
    points = np.asarray(plt.ginput(2))

    # Close plot
    plt.close()
    
    # Get start and end 'x' coordinates
    startX = points[0][0]
    endX = points[1][0]

    # Define x-bounds
    xBounds = (startX, endX)

    # For the y-bounds, mask spectrum and find maximum value
    mask = (bins > startX)*(bins < endX)
    masked_spectrum = spectrum[mask]
    peak_height = np.max(masked_spectrum)

    # Define y-bounds
    startY = (-1)*peak_height*0.2
    endY = peak_height*1.2
    yBounds = (startY, endY)

    # Plot zoomed in area
    plotArgs['title'] = f'{element}: Zoomed in'
    plot_data(bins, spectrum, plotArgs, xBounds=xBounds, yBounds=yBounds)

    ## Zooming further into the region for higher precision
    # Get input points from user (click 2 points on the display)
    points = np.asarray(plt.ginput(2))

    # Close plot
    plt.close()

    # Get start and end 'x' coordinates 
    startX = points[0][0]
    endX = points[1][0]

    ## Obtain fit parameters for the Gaussian fit
    width = endX - startX
    extraR = 0.2

    print(f'Fitting region from bin {startX:.2f} to find {endX:.2f}.\n')

    ## Plotting final fit over region
    plotArgs['title'] = f'{element}: Gaussian fit to spectral line.'
    plot_data(bins, spectrum, plotArgs, xBounds=xBounds, yBounds=yBounds)

    # Masking the bins and calibrated spectrum
    mask = (bins > startX)*(bins < endX)
    masked_bins = bins[mask]
    masked_spectrum = spectrum[mask]

    # Fit the Gaussian to the region of interest (see gaussian_fit.py)
    ## TODO: Return here after gauss.py is done and create a plot

    ### Return fit parameters
    return 

### Prints out spectrum information from NIST to select peaks
def nist_data():
    
    ## Print header
    print("--------------------------------------")
    print(f"index | energy ({units:3}) | relative counts")
    print("--------------------------------------")

    ## Go through every coordinate
    for i, coord in enumerate(coords):

        # Print coordinate with index
        print(f"{i+1:5} | {coords[i][0]:12} | {coords[i][1]}")
    
    print("--------------------------------------")


