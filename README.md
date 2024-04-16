# STEAM Spectrometer Calibration

This project contains Python code designed for calibration of Amptek's soft X-Ray spectrometer and hard X-Ray spectrometer. The code provides the following functionalities:

- **Create a calibration curve:** Generate a calibration curve based on the input data.
- **Calibrate spectra:** Apply the calibration curve to raw spectra to obtain calibrated spectra.
- **Determine the resolution of the spectrometer:** Calculate the resolution of the spectrometer based on the calibrated spectra.
- **Determine the response of the spectrometer:** Evaluate the response of the spectrometer across different energy levels.


## Table of Contents
1. [About the Project](#introduction)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
	- [Prerequisites](#prerequisites)
	- [Installation](#installation)
4. [Calibration Procedure](#calibration-procedure)
	- [Creating a Calibration Curve](#creating-calibration-curve)
	- [Calibrate Spectra](#calibrate-spectra)
	- [Determine Resolution of Spectrometer](#determine-resolution-of-spectrometer)
	- [Determine Response of Spectrometer](#determine-response-of-spectrometer)
5. [Contributing](#contributing)
6. [License](#license)


## About the Project
This code was written to calibrate STEAM's flight model (FM) spectrometers. These tests aimed to calibrate the gain and offset, response, and resolution of two off-the-shelf X-Ray spectrometers, one soft X-Ray spectrometer ([Amptek X-123 SDD](https://www.amptek.com/internal-products/x-123-complete-x-ray-spectrometer)), 
and one hard X-Ray spectrometer ([Amptek X-123 CdTe](https://www.amptek.com/internal-products/x-123-cdte-complete-x-ray-gamma-ray-spectrometer-with-cdte-detector)). Calibration data is derived from the well-known emission lines from calibrated radioisotope sources at various energies within the range of 0-70 keV (STEAM’s region of interest). The radioisotopes and their activities are the following: Am-241 at 1 μCi, Ba-133 at 10 μCi, Cd-109 at 10 μCi, Fe-55 at 100 μCi, and Zn-65 at 10 μCi. 

The gain and offset of the spectrometer were determined using the built-in channels during data analysis. The resolution was determined by the peak widths of the emission lines, while the response was determined by the peak counts. The efficiency of the spectrometer was determined by combining the response and the activity of the isotope.

For more information on how STEAM's spectrometers were calibrated, see the [STEAM Spectrometer Calibration Report](link)  

## Project Structure

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
├───environment
│   ├───environment.yml
│   ├───requirements.txt
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

- `data`: This directory contains all the data files
	- `calibration`: contains the raw calibration data
	- `results`: contains the results of the calibration
		- `cPoints`: contains the calibration points
		- `cCurves`: contains the calibration curve results
		- `resolution`: contains the resolution results
		- `response`: contains the response results

- `environment`: This directory contains the environment files
	- `environment.yml`: contains the environment file for Anaconda
	- `requirements.txt`: contains the environment file for pip

- `utils`: This directory contains various tools written in Python for analyzing the calibration data
	- `classes.py`: contains classes to store data
	- `calibrate.py`: contains functions for calibrating the spectrometer
	- `files.py`: contains functions for file handling
	- `gaussian_fit.py`: contains functions for fitting Gaussian curves to data
	- `llsfit.py`: contains functions for linear least-square fitting
	- `plot.py`: contains functions for plotting data
	- `spectrum.py`: contains functions for reading spectrum data

- `main.py`: This is the main script for running the project

The data used to characterize the spectrometer was collected using both Amptek's DPPMCA software and STEAM's ground software (GSW). The data from Amptek is saved as a ```.mca``` converted to a raw ```.txt``` file and the data from STEAM is initially saved as a ```.csv``` file. STEAM's GSW saves the cumulative spectrum as the last line of the ```.csv``` file. The cumulative spectrum is the sum of all the spectra collected during the run. The cumulative spectrum is used for calibration.

Finally, all commands are run from the ```main.py``` file. This file contains the main function that calls all the other functions in the ```utils``` folder.

## Environment Setup
### Prerequisites

- Visual Studio Code
- Python 3.11.3 or higher
- pip (Python package manager)

### Installation

This project uses the following Python packages:

- matplotlib
- mpl_point_clicker
- numpy
- os
- pandas
- pathlib
- re
- scipy

To install these packages, use the provided environment file by running following steps:

**1.** Install Visual Studio Code (VS Code): [VS Code Download](https://code.visualstudio.com/download)

**2.** Install Python 3.11.3 or higher: [Python Download](https://www.python.org/downloads/)

**3.** In VS Code, install the Python extension and Python interpretor: 
- Python Extension
	- [Visual Studio Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
	- [Anaconda (Recommended)](https://www.anaconda.com/products/individual)
- Python Interpretor for VS code:
	- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)

**4.** Navigate to the project directory: `cd ..\spectrometer_calibration\src`

**5.** Install dependencies, if using:

- 5a. Using Anaconda (recommended)
	- Open Anaconda command prompt. To verify if the Anaconda Prompt is open, type `conda` and press ENTER. You should get a help message that explains usage for conda
	- Create environment and install packages from environment.yml file:
	`conda env create -f environment.yml`
	- Activate environment: `conda activate myenv`
	
	To check if the environment is active, type `conda env list` and press ENTER. You should see a list of environments, and the active environment should be marked with an asterisk (*).

- 5b. Using pip: 
	- Open command prompt. 
	- Create environment: `python -m venv env`
	- Activate environment: `source myenv/bin/activate`
	- Install packages from requirements.txt file: `pip install -r environment/requirements.txt`


	*venv* is a module that comes with Python 3.3 and later versions, so you do not need to install it separately. It is already available on both OS and Windows, as long as you have a compatible Python version. You can check your Python version by running `python --version` in a terminal or command prompt. If you get a message saying that the command was not found or something similar, you’ll need to install Python. If you get a version number (e.g. Python 3.11.2), you’re good to go.

**6.** Run the project using the commands outlined in the [Calibration Procedure](#calibration-procedure) section.

## Calibration Procedure (Usage)

### Creating a Calibration Curve
This section outlines the steps to perform a Gaussian fit on each desired peak within the spectrum, export the results, and create a calibration curve. 

Before starting, ensure you have the following:
- **Spectra files:** These can be in the form of .csv or .txt files, and can be a single file or a folder containing multiple files.
	- [Example .csv data file](example_files\SDD_Fe_55_Data_Example.csv) (from GSW)
	- [Example .txt data file](example_files\SDD_Fe_55_Data_Example.txt) (from DPPMCA)

Follow the steps below to create a calibration curve.:

**Step 1.** Enter the command in terminal to fit each spectra. (See example data files above)
- Command: ```python main.py -cp```

	or,  ```python main.py --calibrate_points```

**Step 2:** Enter file(s) or folder(s) location of data to use for calibration.
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data```

**Step 3:** Follow prompt to perform a Gaussian fit for each peak.
- [Example calibration points file](example_files/SDD_Fe_55_Calibration_Points.csv)

**Step 4:** Repeat steps 1-3 for each spectrum.

**Step 5:** Enter command in terminal to create calibration curve.
- Command: ```python main.py -cc``` 

	or, ```python main.py --calibration```

**Step 6:** Enter  _all_  files or folder(s) location of calibration points to use for calibration. (See Step 3 for example file)

**Step 7:** Follow prompts.
- [Example of exported calibration curve csv file]
- [Example of exported calibration curve plot]

----
### Calibrate Spectra
This section outlines the steps to calibrae the spectra using the calibration curve. 

Before starting, ensure you have the following:
- **Spectra files:** These can be in the form of .csv or .txt files, and can be a single file or a folder containing multiple files.
	- [Example .csv data file](example_files\SDD_Fe_55_Data_Example.csv) (from GSW)
	- [Example .txt data file](example_files\SDD_Fe_55_Data_Example.txt) (from DPPMCA)
- **Calibration curve file:** This is the results of the calibration curve 
	- Example calibration curve file

Follow the steps below to create a calibration curve.:

**Step 1:** Enter command in terminal to calibrate spectra.
- Command: ```python main.py -c``` 

	or, ```python main.py --calibrate```

**Step 2:** Enter file(s) or folder(s) location of spectrum or spectra to calibrate.
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data```

**Step 3:** Enter file location of calibration curve.
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data\results\cCurves\calibration_curve.csv```

**Step 4:** Follow prompts.
- Example of exported calibrated spectrum .csv file
- Example of exported calibrated spectrum plot

---
### Determine Resolution of Spectrometer

Before continuing, ensure you have the following:
 - **Calibration curve points file:** This file contains all of the calibration points used to make the calibration curve. Created in the [Calibration Curve](#calibration-curve) section.
	- Example calibration points file
 - **Expected Spectrum:** This file can be created using STEAM's IDL code, or your own. From this file needs to contain the expected max counts of the desired energy peaks.

Follow the steps below to determine the detector response.:

**Step 1:** Enter command into terminal to determine the detector response.
- Command: ```python main.py -r``` 

	or, ```python main.py --resolution```

**Step 2:** Enter file location of calibration points file.
	- Example calibration points file

**Step 3:** Enter file location of expected counts file.
	- Example of expected counts file (from STEAM's IDL)

 **Step 4:** Follow prompts.
 	- Example of exported resolution file
  	- Example of exported resolution plot (Measured vs Expected Max Counts)


### Determine Response of Spectrometer
This section outlines the steps to determine the detector response by analyzing the Full Width Half Max (FWHM) and energy of the calibration points. The results are saved to a CSV file and a plot of FWHM vs Energy is displayed.

Before continuing, ensure you have the following:
- **Calibration curve points file:** This file contains all of the calibration points used to make the calibration curve. Created in the [Calibration Curve](#calibration-curve) section.
	- Example calibration points file

Follow the steps below to determine the detector response.:

**Step 1:** Enter command into terminal to determine the detector response.
- Command: ```python main.py -rp``` 

	or,  ```python main.py --response```

**Step 2:** Enter file location of calibration curve. (Results of calibration curve _curve.csv file)
- Example File Location: ```C:\repo\steam_science\spectrometer_calibration\src\data\results\cPoints```

**Step 3:** Follow prompts.
- Example of exported response .csv file
- Example of exported response plot


## Contributing


## License
