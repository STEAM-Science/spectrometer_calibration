import pathlib
import os
import re

def load_folder():
	"""
	Prompts the user to enter a folder path and returns a list of filenames in the folder.

	Returns:
	- A list of filenames in the folder specified by the user.

	Raises:
	- ValueError: If the folder path entered by the user is not a directory.
	"""
	filenames = []

	user_input = input("Enter folder path (blank to quit): ")
	while user_input != "":
		folder = pathlib.Path(user_input)
		if folder.is_dir():
			for file in folder.iterdir():
				filenames.append(file)
				print("Added", file, "to list of files")
		else:
			raise ValueError("Folder path is not a directory")
		user_input = input("Enter folder path (blank to quit): ")

	return filenames


def load_files():
	"""
	Prompts the user to enter file paths and returns a list of pathlib.Path objects representing the files.

	Returns:
	list: A list of pathlib.Path objects representing the files.
	"""
	filenames = []

	user_input = input("Enter file path (blank to quit): ")
	while user_input != "":
		file = pathlib.Path(user_input)
		filenames.append(file)
		user_input = input("Enter file path (blank to quit): ")
	return filenames

def create_csv(data, name, folder_path):
	"""
	Creates a CSV file with the outputs from the Gaussian fit.

	"""
	print('Creating CSV file...')
	
	## Check if name is valid
	if re.search(r'[\\/:*?"<>|]', name):
		print("Invalid file name. Please enter a valid name.")

	## Save data to CSV file
	file_path = folder_path + f"\{name}.csv"

	data.to_csv(file_path, index=False, encoding='utf-8')
		

def get_folder_path():
	user_input = input("Where would you like to save the file?: ")

	return user_input


def create_image():
	"""
	Creates a PNG file with the outputs from the Gaussian fit.

	"""

def find_file_path(name, path):
	"""
	file path and returns the path as a string.

	Returns:
	- A string representing the file path.
	"""
	for root, dirs, files in os.walk(path):
		if name in files:
			return os.path.join(root, name)