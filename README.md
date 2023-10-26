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
More detailed here?

## Environment Setup
### Prerequisites

- Visual Studio Code
- Python 3.11.3

### Installation


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
│   ├───files.py
│   ├───read_spectrum.py
│   ├───plot.py
│   ├───gaussian_fit.py
│   ├───llsfit.py (linear least-square fitting)
│   ├───calibrate.py
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
	- `files.py`: contains functions for file handling
	- `read_spectrum.py`: contains functions for reading spectrum data
	- `gaussian_fit.py`: contains functions for fitting Gaussian curves to data
	- `llsfit.py`: contains functions for linear least-square fitting
	- `plot.py`: contains functions for plotting data
	- `calibrate.py`: contains functions for calibrating the spectrometer

- `main.py`: contains the main script for running the project

All data used to characterize the spectrometer is stored in the ```calibration``` folder under data. This data was collected using both Amptek's DPPMCA software and STEAM's software. The data from Amptek is saved as a ```.mca``` converted to a raw ```.txt``` file and the data from STEAM is initially saved as a ```.csv``` file. To further understand how we use this data to characterize the spectrometer, see the [Calibration Procedure](#calibration-procedure) section ***or see our Report?**

Finally, all commands are run from the ```main.py``` file. This file contains the main function that calls all the other functions in the ```utils``` folder.

## Calibration Procedure (Usage)


## Contributing


## License
