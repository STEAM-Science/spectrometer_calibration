### Importing useful packages
import numpy as np

### Unweighted fit
### Function to calculate delta for an unweighted fit
def delta_unweighted(x, y):
    return len(x)*np.sum(x*x) - (np.sum(x))**2

### Function to calculate slope for an unweighted fit
def m_unweighted(x, y):
    delta = delta_unweighted(x, y)
    return (len(x)*np.sum(x*y)-np.sum(x)*np.sum(y))/delta

### Function to calculate intercept for an unweighted fit
def b_unweighted(x, y):
    delta = delta_unweighted(x, y)
    return (np.sum(x*x)*np.sum(y)-np.sum(x)*np.sum(x*y))/delta

### Function to caluclate sigma y for an unweighted fit
def sig_y_unweighted(x, y):
    m = m_unweighted(x, y)
    b = b_unweighted(x, y)
    return np.sqrt(np.sum((y-m*x-b)**2)/(len(x)-2))

### Function to calculate uncertainty on slope for an unweighted fit
def sig_m_unweighted(x, y):
    delta = delta_unweighted(x, y)
    sig_y = sig_y_unweighted(x, y)
    return sig_y*np.sqrt(len(x)/delta)

### Function to calculate uncertainty on intercept for an unweighted fit
def sig_b_unweighted(x, y):
    delta = delta_unweighted(x, y)
    sig_y = sig_y_unweighted(x, y)
    return sig_y*np.sqrt(np.sum(x*x)/delta)


### Get numerical input
def getNumericalInput(text):

    ## Define temporary variable
    num = 'temp'

    ## Check if number inputted
    while not num.isnumeric():

        # Get input
        num = input(text)

        # If this is not a numerical value
        if not num.isnumeric():

            print('Input a number!')

    ## Return input as float
    return float(num)