import pathlib

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

def create_csv(spectral_data):
	"""
	Creates a CSV file with the outputs from the Gaussian fit.

	"""

def create_image():
	"""
	Creates a PNG file with the outputs from the Gaussian fit.

	"""