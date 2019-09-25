class Fruit:
	"""
	Used to hold and handle Defect objects
	"""

	def __init__(self, index, load_path, defects_thresholds=[160]):
		"""
		Instantiates the Fruit object
		"""

		self.index = index
		self.defects = None