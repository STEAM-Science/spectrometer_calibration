# STEAM Spectrometer Calibration Code
Python code to create calibration curves for Amptek CdTe (hard) and FastSDD (soft) X-ray spectrometers. This code has the following functions:

1. Obtaining calibration points
2. Creating calibration curves
3. Displaying calibrated spectra
4. Exporting `.csv` files of calibrated spectra

Data is stored in the STEAM shared Google Drive [here](https://drive.google.com/drive/folders/1rcfaDmzqOL7TGXesyyXhKRsfRpkerUmS?usp=share_link).

## 1. Obtaining calibration points

Calibration points can be obtained by fitting known spectral peaks to their corresponding energies. This is done using the `--cPoints` flag while running `main.py`.

A demonstration of this procedure can be done by running

`python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cPoints`

Here, the source spectrum being used to obtain the calibration points is specified at `spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt` with the flag `--src`.

Once the script is run, the program will display the uncalibrated spectrum. Count the number of peaks of known energy that will be fit (integer input).

Next, the program will iterate over every desired peak. The uncalibrated spectra will be displayed once again and a table of expected peaks from the NIST database will be printed. Choose the peak and enter its NIST index. Note that it is easiest to go in a descending order of peak counts.

Now, the plot will be displayed and the user shall be prompted to zoom into the desired peak. Do this by clicking to the left of and to the right of a peak. It is, in general, better to select points sufficiently far from the peak.

The program will zoom into the desired range and request a range to fit a Gaussian curve. Perform this fit by selecting points on either side of the peak in a manner similar to the previous step, however, ensure that no other peaks are contained in the range.

This will result in the acquisition of a single point for the calibration curve. Repeat this process for all known peaks (the program will automatically iterate over the number specified at the start). At the end of this task, the points will be saved with errorbars at `cPoints\<subfolder>\<filename>_curve.csv`.

## 2. Creating calibration curves

Calibration curves can be created by fitting a curve mapping bins of known spectral peaks to their corresponding energies. This is done using the `--calibrate` flag while running `main.py`.

A demonstration of this procedure can be done by running

`python main.py --calibrate`

Once the script is run, the program will ask for inputs of calibration points. Input addresses here one at a time.

Once all points `.csv` files are specified, the program will calculate the calibration curve. The user will be asked to input a name for the output file

The curve will be displayed and saved with fit points and will be saved at `cCurves\<filename>.csv`.

## 3. Displaying calibrated spectra

Spectra measured by the Amptek spectrometers stored in raw text (`.txt`) format can be displayed as calibrated spectra using the `--display` flag.

A demonstration of this procedure can be done by running

`python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cSrc cCurves/demo/2022_12_02_CdTe_Zn_01_no_purge_curve.csv --display`

Here, the source spectrum being used to create the calibration curve is specified at `spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt` with the flag `--src`. The calibration curve used to calibrate this spectrum is specified at `cCurves/demo/2022_12_02_CdTe_Zn_01_no_purge_curve.csv` with the flag `--cSrc`.

Once this script is run, the user will be asked whether they want specific bounds for the x-axis. If yes, these bounds must be provided in units of keV.

Next, the user will be asked whether to save the resultant plot. If yes, the plot will be saved at `plots\<subfolder>\<filename>.png`.

## 4. Exporting `.csv` files of calibrated spectra

Spectra measured by the Amptek spectrometers stored in raw text (`.txt`) format can be calibrated and saved as a dataframe using the `--csv` flag.

A demonstration of this procedure can be done by running

`python main.py --src spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt --cSrc cCurves/demo/2022_12_02_CdTe_Zn_01_no_purge_curve.csv --csv`

Here, the source spectrum being used to create the calibration curve is specified at `spectra\demo\2022_12_02_CdTe_Zn_01_no_purge.txt` with the flag `--src`. The calibration curve used to calibrate this spectrum is specified at `cCurves/demo/2022_12_02_CdTe_Zn_01_no_purge_curve.csv` with the flag `--cSrc`.

The resultant `.csv` file will be stored at `csvOut\<subfolder>\<filename>.csv`.

## Displaying README.md

Display `README.md` using the following:
> !grip -b README.md

For issues with the code email Ravin Chowdhury at `rach1691@colorado.edu`.