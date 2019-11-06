import tifffile
from libxmp import XMPFiles, consts
import ast
from src.fruit.defect import Defect
from skimage.measure import label, regionprops
from uuid import uuid4

class Fruit:
	"""
	Used to hold and handle Defect objects
	"""

	def __init__(self, index, load_path, defects_thresholds=[160]):
		"""
		Instantiates the Fruit object

		Parameters
		----------
		load_path : str
			load path of the fruit's shots
		defects_thresholds : list
			list of thresholds for labeling
		"""

		self.index = index

		self.shots = self.load(load_path, index, defects_thresholds)
		self.shots_tot = len(self.shots)
		self.shots_keys = [key for key, vals in self.shots.items()]
		self.shots_to_analyze = [shot for shot, defects_list in self.shots.items() if len(defects_list)]
		self.shots_analyzed = []

		self.is_analizable = len(self.shots_to_analyze) > 1	

		self.defects_tot = sum([len(self.shots[index]) for index in self.shots_to_analyze])

		self.current_shot = None
		self.d_index = None
		if self.shots_to_analyze:
			self.setup()
		
	def __iter__(self):
		return self

	def __next__(self):
		"""
		Return the next defect to be analyzed and updates all the counters

		Returns
		-------
		defect : Defect
			next defect to be analyzed
		"""

		if not self.d_index < len(self.shots[self.current_shot]):
			try:
				self.temp_shot = self.shots_to_analyze.pop(0)
			except:
				raise StopIteration
			self.d_index = 0
			self.shots_analyzed.append(self.current_shot)
			self.current_shot = self.temp_shot
			self.shots_index = self.shots_keys.index(self.current_shot)
			
		defect = self.shots[self.current_shot][self.d_index]
		self.defects_index += 1
		self.d_index += 1
		return defect

	def load(self, load_path, fruit_index, defects_thresholds):
		"""
		Loads shots and answers (indices of defects on the fruit)

		Parameters
		----------
		load_path : str
			load path of the fruit's shots
		defects_thresholds : list
			list of thresholds for labeling

		Returns
		-------
		defects : list
			list of defects divided in sublists (shots)
		"""

		name = load_path + "{0}.tiff".format(fruit_index)
		shots = tifffile.TiffFile(name).asarray()
		xmpfile = XMPFiles(file_path=name, open_forupdate=True).get_xmp()
		answers = ast.literal_eval(xmpfile.get_property(consts.XMP_NS_DC, "description[1]"))

		shots_dict = {}
		for i, (shot, ans) in enumerate(zip(shots, answers)):
			thresholds = shot < defects_thresholds[0]
			labels = label(thresholds)
			properties = regionprops(labels, coordinates="rc")

			defects = [Defect("{0}_{1}".format(fruit_index, i), defect_index, shot.shape, props)\
						for defect_index, props in zip(ans, properties)]
			shots_dict["shot_{0}".format(i)] = defects

		return shots_dict

	def setup(self):

		if self.shots_to_analyze:
			self.current_shot = self.shots_to_analyze.pop(0)
			self.d_index = 0
			self.defects_index = 0

		for defect in self.shots[self.current_shot]:
			defect.uuid = uuid4()
			self.__next__()