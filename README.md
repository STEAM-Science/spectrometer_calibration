# STEAM Spectrometer Calibration

Python code to create calibration curves for Amptek's soft X-Ray spectrometer ([Amptek X-123 SDD](https://www.amptek.com/internal-products/x-123-complete-x-ray-spectrometer)) and hard X-Ray spectrometer ([Amptek X-123 CdTe](https://www.amptek.com/internal-products/x-123-cdte-complete-x-ray-gamma-ray-spectrometer-with-cdte-detector)). This code has the following functions:

1. Creating calibration curves
2. Displaying calibrated spectra
3. Exporting `.csv` files of calibrated spectra


## Environment Setup
### Tools

- Visual Studio Code
- Python 3.11.3


### Project Structure

```
root
├───data (contains all the data files)
│   ├───calibration
│   ├───results
├───utils
│   ├───read_spectra.py
│   ├───output_spectra.py
│   ├───gaussian_fit.py
│   ├───llsfit.py (linear least-square fitting)
main.py
```
All data used to characterize the spectrometer is stored in the ```calibration``` folder under data. This data was collected using both Amptek's DPPMCA software and STEAM's software. The data from Amptek is saved as a ```.mca``` converted to a raw ```.txt``` file and the data from STEAM is initially saved as a ```.csv``` file, also converted to a ```.txt``` file. To further understand how we use this data to characterize the spectrometer, see the [Calibration Procedure](#calibration-procedure) section ***or see our Report?**

The ```utils``` folder contains tools for analyzing the calibration data. The ```read_spectra.py``` file contains functions for reading the uploaded data and the ```output_spectra.py``` file contains functions for exporting the calibrated spectra as a csv and displaying the results. The ```gaussian_fit.py``` file contains functions for fitting the calibration data to a Gaussian curve. The ```llsfit.py``` file contains functions for fitting the calibration data to a linear curve.

Finally, all commands are run from the ```main.py``` file. This file contains the main function that calls all the other functions in the ```utils``` folder.