# import the necessary packages
import h5py
import os
import numpy as np


class HDF5DatasetWriter:
	def __init__(self, dims, outputPath, label_dim=1, label_dtype="int", input_dtype='float', dataKeys=("X", "y"), bufSize=1000):
		# check to see if the output path exists, and if so, raise
		# an exception
		if os.path.exists(outputPath):
			raise ValueError("The supplied `outputPath` already "
				"exists and cannot be overwritten. Manually delete "
				"the file before continuing.", outputPath)

		# open the HDF5 database for writing and create two datasets:
		# one to store the images/features and another to store the
		# class labels
		self.db = h5py.File(outputPath, "w")
		self.X = self.db.create_dataset(dataKeys[0], dims, dtype=input_dtype)
		self.y = self.db.create_dataset(dataKeys[1], (dims[0], label_dim), dtype=label_dtype)

		# store the buffer size, then initialize the buffer itself
		# along with the index into the datasets
		self.bufSize = bufSize
		self.buffer = {"X": [], "y": []}
		self.idx = 0

	def add(self, rows, labels):
		# add the rows and labels to the buffer
		self.buffer["X"].extend(rows)
		self.buffer["y"].extend(labels)

		# check to see if the buffer needs to be flushed to disk
		if len(self.buffer["X"]) >= self.bufSize:
			self.flush()

	def flush(self):
		# write the buffers to disk then reset the buffer
		i = self.idx + len(self.buffer["X"])
		self.X[self.idx:i] = self.buffer["X"]
		self.y[self.idx:i] = np.array(self.buffer["y"]).reshape(self.y[self.idx:i].shape)
		self.idx = i
		self.buffer = {"X": [], "y": []}

	def storeClassLabels(self, classLabels):
		# create a dataset to store the actual class label names,
		# then store the class labels
		dt = h5py.special_dtype(vlen=str) # `vlen=unicode` for Py2.7
		labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
		labelSet[:] = classLabels

	def close(self):
		# check to see if there are any other entries in the buffer
		# that need to be flushed to disk
		if len(self.buffer["X"]) > 0:
			self.flush()

		# close the dataset
		self.db.close()