### Read uploaded spectrum file and display it

### Importing useful packages
import os
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
from gauss import *
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

            # go to .csv function read
            csv_read(filepath)
      
        else:
            raise ValueError('File format not .csv!')

    ## If file is a .txt continue to the next step
    elif user_file_case == 'txt':

        print(f'Reading file: {filepath}')

        # Check file format and continue
        if ext == '.txt':

            # go to .txt function read
            txt_read(filepath)

        else:
            raise ValueError('File format not .txt!')

    else:
            
        raise ValueError('File format not supported!')

### Read .csv file and save spectrum data
def csv_read(filepath):

    ## Read file
    df = pd.read_csv(filepath)

    ## Get data (last row of .csv file) and convert to array
    spectral_data = df.iloc[-1].to_numpy()

    ### Return data
    return spectral_data

### Read .txt file and save spectrum data
def txt_read(filepath):

    ## Read file
    df = pd.read_table(filepath)

    return

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
    
    return

### Prints out spectrum information from NIST
def nist_data():
    return