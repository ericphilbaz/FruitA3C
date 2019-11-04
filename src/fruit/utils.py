import numpy as np

def log10_trasform(values):
	return -np.multiply(np.sign(values), np.log10(np.abs(values)))

def sigmoid(x):
	return 1 / (1 + np.exp(-x))