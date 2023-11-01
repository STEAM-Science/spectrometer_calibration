"""
files.py - A module for loading and processing files.

This module provides functions for loading a folder of files and processing them. 
It also provides functions for saving data to files.

Functions:
- load_folder(): Prompts the user for a folder path and returns a list of all files in the folder.
- load_files(): Prompts the user for a file path and returns a list of all files.
- get_folder_path(): Prompts the user for a folder path and returns it as a string.
- find_file_path(name, path): Searches for file with the given name in the specified directory and its subdirectories.
- create_csv(data, filename, output_path): Creates a CSV file from a pandas DataFrame and saves it to a specified output path.
- create_image(filename, output_path): Creates an image from a matplotlib plot and saves it to a specified output path.
"""

### Import libraries
import pathlib
import os
import re
import matplotlib.pyplot as plt

### Reads uploaded folder and stores all files in a list
def load_folder():

	## List of filenames
	filenames = []

	## Get folder path from user
	user_input = input("Enter folder path (blank to quit): ")

	## Replacing double quotes with nothing from user input
	## When copying from Windows Explorer, the path sometimes copied with double quotes, which would create an 
	## invalid path.
	user_input = user_input.replace('"', '')

	## If the user input is not blank,
	while user_input != "":

		# Finds folder path
		folder = pathlib.Path(user_input)

		# If the folder path is a directory,
		if folder.is_dir():

			# Add all files in the folder to the list of filenames
			for file in folder.iterdir():
				filenames.append(file)
				print("Added", file.stem, "to list of files")
		else:

			raise ValueError("Folder path is not a directory")

		user_input = input("Enter folder path (blank to quit): ")

	### Returns list of filenames (paths) within the folder
	return filenames

### Reads uploaded file(s) and stores all files in a list
def load_files():

	filenames = []

	user_input = input("Enter file path (blank to quit): ")

	## Replacing double quotes with nothing from user input
	## When copying from Windows Explorer, the path sometimes copied with double quotes, which would create an 
	## invalid path.
	user_input = user_input.replace('"', '')

	## If user input is not blank,
	while user_input != "":

		# Finds file path
		file = pathlib.Path(user_input)

		# Add it to the list of filenames
		filenames.append(file)

		print("Added", file, "to list of files")

		user_input = input("Enter file path (blank to quit): ")
	
	### Returns list of filenames (paths)
	return filenames

### Asks user for folder path and returns it as a string
def get_folder_path():

	user_input = input("Where would you like to save the file? (Location Path): ")

	## Replacing double quotes with nothing from user input
	## When copying from Windows Explorer, the path sometimes copied with double quotes, which would create an 
	## invalid path.
	user_input = user_input.replace('"', '')
	
	return user_input

### Searches for file with the given name in the specified directory and its subdirectories.
def find_file_path(name, path):
	
		for root, dirs, files in os.walk(path):
				if name in files:
						return os.path.abspath(os.path.join(root, name))
		return 

### Creates a csv with inputted data and saves it to the specified location
def create_csv(data, name, folder_path):
	
	print('\nCreating CSV file...')

	## Replacing double quotes with nothing from user input
	## When copying from Windows Explorer, the path sometimes copied with double quotes, which would create an 
	## invalid path.
	folder_path = folder_path.replace('"', '')
	
	## Check if name is valid
	if re.search(r'[\\/:*?"<>|]', name):
		print("Invalid file name. Please enter a valid name.")

		# Asks user for a new name
		name = input("Enter file name: ")

	## Replacing spaces with underscores from user input
	name = name.replace(' ', '_')

	## Save data to CSV file
	file_path = folder_path + f"\{name}.csv"

	data.to_csv(file_path, index=False, encoding='utf-8')

	print(f"Saved as {name} to {folder_path}.")


### Creates an image with inputted data and saves it to the specified location
def create_image(name, folder_path):

	print('\nSaving image...')

	## Replacing double quotes with nothing from user input
	## When copying from Windows Explorer, the path sometimes copied with double quotes, which would create an 
	## invalid path.
	folder_path = folder_path.replace('"', '')

	## Check if name is valid
	if re.search(r'[\\/:*?"<>|]', name):
		print("Invalid file name. Please enter a valid name.")

		# Asks user for a new name
		name = input("Enter file name: ")

	## Replacing spaces with underscores from user input
	name = name.replace(' ', '_')

	## Save data to PNG file
	file_path = folder_path + f"\{name}.png"

	plt.savefig(file_path, bbox_inches='tight', dpi=300)

	print("Save successful!")
	print(f"Image saved as {name}_plot.png.")