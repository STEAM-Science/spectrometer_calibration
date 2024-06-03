class DataProcessor:
	def __init__(self, filename, data):
		self.filename = filename
		self.data = data

class CombinedIsotopeCalibrationData:
	def __init__(self):
		self.mu = []
		self.E = []
		self.sigma = []
		self.int_counts = []
		self.max_counts = []
	
	def add_mus(self, isotope, mu):
		self.mu.append((isotope, mu))

	def add_Es(self, isotope, E):
		self.E.append((isotope, E))

	def add_sigmas(self, isotope, sigma):
		self.sigma.append((isotope, sigma))

	def add_int_counts(self, isotope, int_counts):
		self.int_counts.append((isotope, int_counts))

	def add_max_counts(self, isotope, max_counts):
		self.max_counts.append((isotope, max_counts))

		# CombinedItosomeCalibrationData.mu. --> [(isotope, mu), (isotope, mu), ...]
		# array[][1] --> mu
