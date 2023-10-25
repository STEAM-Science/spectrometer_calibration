### Importing useful packages
import numpy as np

### Linear Least Square Fit

### Unweighted fit
## Function to calculate delta for an unweighted fit
def delta_unweighted(x, y):
    return len(x)*np.sum(x*x) - (np.sum(x))**2

## Function to calculate slope for an unweighted fit
def m_unweighted(x, y):
    delta = delta_unweighted(x, y)
    return (len(x)*np.sum(x*y)-np.sum(x)*np.sum(y))/delta

## Function to calculate intercept for an unweighted fit
def b_unweighted(x, y):
    delta = delta_unweighted(x, y)
    return (np.sum(x*x)*np.sum(y)-np.sum(x)*np.sum(x*y))/delta

## Function to caluclate sigma y for an unweighted fit
def sig_y_unweighted(x, y):
    m = m_unweighted(x, y)
    b = b_unweighted(x, y)
    return np.sqrt(np.sum((y-m*x-b)**2)/(len(x)-2))

## Function to calculate uncertainty on slope for an unweighted fit
def sig_m_unweighted(x, y):
    delta = delta_unweighted(x, y)
    sig_y = sig_y_unweighted(x, y)
    return sig_y*np.sqrt(len(x)/delta)

## Function to calculate uncertainty on intercept for an unweighted fit
def sig_b_unweighted(x, y):
    delta = delta_unweighted(x, y)
    sig_y = sig_y_unweighted(x, y)
    return sig_y*np.sqrt(np.sum(x*x)/delta)

### Weighted fit
## Function to find weights
def weights(x, y, sigmas):
    return 1/(sigmas**2)

## Function to find delta (weighted)
def delta_weighted(x, y, sigmas):
    w = weights(x, y, sigmas)
    return np.sum(w)*np.sum(w*x*x)-(np.sum(w*x))**2

## Function to find slope of weighted fit
def m_weighted(x, y, sigmas):
    w = weights(x, y, sigmas)
    delta = delta_weighted(x, y, sigmas)
    return (np.sum(w)*np.sum(w*x*y)-np.sum(w*x)*np.sum(w*y))/delta

## Function to find intercept of weighted fit
def b_weighted(x, y, sigmas):
    w = weights(x, y, sigmas)
    delta = delta_weighted(x, y, sigmas)
    return (np.sum(w*x*x)*np.sum(w*y)-np.sum(w*x)*np.sum(w*x*y))/delta

## Function to find uncertainty on slope of weighted fit
def sig_m_weighted(x, y, sigmas):
    w = weights(x, y, sigmas)
    delta = delta_weighted(x, y, sigmas)
    return np.sqrt(np.sum(w)/delta)

## Function to find uncertainty on intercept of weighted fit
def sig_b_weighted(x, y, sigmas):
    w = weights(x, y, sigmas)
    delta = delta_weighted(x, y, sigmas)
    return np.sqrt(np.sum(w*x*x)/delta)