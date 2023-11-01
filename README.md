# STEAM Spectrometer Calibration

Python code to create calibration curves for Amptek's soft X-Ray spectrometer ([Amptek X-123 SDD](https://www.amptek.com/internal-products/x-123-complete-x-ray-spectrometer)) and hard X-Ray spectrometer ([Amptek X-123 CdTe](https://www.amptek.com/internal-products/x-123-cdte-complete-x-ray-gamma-ray-spectrometer-with-cdte-detector)). This code has the following functions:


## Table of Contents
1. [About the Project](#introduction)
2. [Environment Setup](#environment-setup)
	- [Prerequisites](#prerequisites)
	- [Installation](#installation)
3. [Calibration Procedure](#calibration-procedure)
4. [Contributing](#contributing)
5. [License](#license)


## About the Project
This code was written to calibration STEAM's flight model (FM) spectrometers. These tests aimed to calibrate the gain and offset, response, and resolution of two off-the-shelf (OTS) X-Ray spectrometers, one soft X-Ray spectrometer ([Amptek X-123 SDD](https://www.amptek.com/internal-products/x-123-complete-x-ray-spectrometer)), 
and one hard X-Ray spectrometer ([Amptek X-123 CdTe](https://www.amptek.com/internal-products/x-123-cdte-complete-x-ray-gamma-ray-spectrometer-with-cdte-detector)). Calibration data is derived from the well-known emission lines from calibrated radioisotope sources at various energies within the range of 0-70 keV (STEAM’s region of interest). The radioisotopes and their activities are the following: Am-241 at 1 μCi, Ba-133 at 10 μCi, Cd-109 at 10 μCi, Fe-55 at 100 μCi, and Zn-65 at 10 μCi. 

The gain and offset of the spectrometer were determined using the built-in channels during data analysis. The resolution was determined by the peak widths of the emission lines, while the response was determined by the peak counts. The efficiency of the spectrometer was determined by combining the response and the activity of the isotope.

For more information on how STEAM's spectrometers were calibrated, see the [STEAM Spectrometer Calibration Report](link)  

## Environment Setup
### Prerequisites

- Visual Studio Code
- Python 3.11.3 or higher
- pip (Python package manager)

### Installation
	1. Navigate to the project directory: `cd project path` or `cd ..\spectrometer_calibration\src`
	2. Install dependencies: `pip install -r requirements.txt`
	3. Run the project:  `python main.py`

### Project Structure

```
root
├───data (contains all the data files)
│   ├───calibration
│   │   ├───processed
│   ├───results
│   │   ├───cPoints
│   │   ├───cCurves
│   │   ├───resolution
│   │   ├───response
├───utils
│   ├───calibrate.py
│   ├───classes.py
│   ├───files.py
│   ├───gaussian_fit.py
│   ├───llsfit.py (linear least-square fitting)
│   ├───plot.py
│   ├───spectrum.py
main.py
```

- `data`: contains all the data files
	- `calibration`: contains the raw calibration data
	- `results`: contains the results of the calibration
		- `cPoints`: contains the calibration points
		- `cCurves`: contains the calibration curve results
		- `resolution`: contains the resolution results
		- `response`: contains the response results

- `utils`: contains tools for analyzing the calibration data
	- `classes.py`: contains classes to store data
	- `calibrate.py`: contains functions for calibrating the spectrometer
	- `files.py`: contains functions for file handling
	- `gaussian_fit.py`: contains functions for fitting Gaussian curves to data
	- `llsfit.py`: contains functions for linear least-square fitting
	- `plot.py`: contains functions for plotting data
	- `spectrum.py`: contains functions for reading spectrum data

- `main.py`: contains the main script for running the project

This data used to characterize the spectrometer was collected using both Amptek's DPPMCA software and STEAM's ground software. The data from Amptek is saved as a ```.mca``` converted to a raw ```.txt``` file and the data from STEAM is initially saved as a ```.csv``` file. 

Finally, all commands are run from the ```main.py``` file. This file contains the main function that calls all the other functions in the ```utils``` folder.

## Calibration Procedure (Usage)

### Calibration Curve
Perform a Gaussian fit on each desired peak within the spectrum in the spectrum and plot the results. The user will be prompted to enter the location of the data to be calibrated. The data can be a single file or a folder containing multiple files. With the results, the user can then create the calibration curve. Steps are below.

Example data files:
- Example csv file
- Example txt file

**Step 1.** Enter comand in terminal to fit each spectra.
- Command: ```python main.py -cp or python main.py --calibrate_points```

**Step 2:** Enter file(s) or folder(s) location of data to use for calibration.
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data```


**Step 3:** Follow prompts.
- Example of exported file

**Step 4:** Repeat steps 1-3 for each spectrum.

**Step 5:** Enter command in terminal to create calibration curve.
- Command: ```python main.py -cc or python main.py --calibration```

**Step 6:** Enter  _all_  file(s) or folder(s) location of calibration points to use for calibration.
- Example of exported calibration points csv file

**Step 7:** Follow prompts.
- Example of exported calibration curve csv file
- Example of exported calibration curve plot

----
### Calibrate Spectra
Calibrate the spectra using the calibration curve. The user will be prompted to enter the location of the data to be calibrated. The data can be a single file or a folder containing multiple files. With the results, the user can then create the calibrated spectra. Steps are below.

**Step 1:** Enter command in terminal to calibrate spectra.
- Command: ```python main.py -c``` or ```python main.py --calibrate```

**Step 2:** Enter file(s) or folder(s) location of spectrum or spectra to calibrate.
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data```

**Step 3:** Enter file location of calibration curve.
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data\results\cCurves\calibration_curve.csv```

**Step 4:** Follow prompts.
- Example of exported calibrated spectrum csv file
- Example of exported calibrated spectrum plot

---
### Resolution
Command: ```python main.py -r``` or ```python main.py --resolution```



### Response
Command: ```python main.py -p``` or ```python main.py --response```




## Contributing


## License
