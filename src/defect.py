class Defect:
	"""
	Defect object find on fruits
	"""

	def __init__(self, shot_name, index, bounding_box, area, shot_sizes):
		"""
		Instantiates Defect objects

		Parameters
		----------
		shot_name : str
			name of the shot
		index : int
			index of the defect (on the fruit)
		bounding_box : array
			bounding limits of the defect
		area : int
			area of the defect
		shot_sizes : array
			sizes of the shot
		"""

		self.shot_name = shot_name
		self.shot_sizes = shot_sizes

		self.ID = None
		self.index = index

		self.x_center = (bounding_box[3] - bounding_box[1])/2
		self.y_center = (bounding_box[2] - bounding_box[0])/2
		self.area = area

	def __sub__(self, defect):
		"""
		Used to return differences between two defects

		Parameters
		----------
		defect : Defect
			other defect

		Returns
			an array of differences, in absolute value, between 0 (different) and 1 (same)
		"""

		delta_x = 1 - np.abs(self.xc - defect.xc)/self.shot_sizes[1]
		delta_y = 1 - np.abs(self.yc - defect.yc)/self.img_sizes[0]
		delta_area = 1 - np.abs(self.area - defect.area)/(self.img_sizes[1]*self.img_sizes[0])

		return np.array([delta_x, delta_y, delta_A]).reshape((1, 3))