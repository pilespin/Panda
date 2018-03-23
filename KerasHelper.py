
from keras.models import model_from_yaml
import numpy as np
import Image
import os

class KerasHelper:
	'Keras IO'
	def __init__():
		'Keras Initialized'

	###########################################################################
	#################################### IO ###################################
	###########################################################################

	@staticmethod
	def save_model(model, path):
		path_yaml = path + ".yaml"
		path_h5 = path + ".h5"

		model_yaml = model.to_yaml()
		with open(path_yaml, "w") as yaml_file:
			yaml_file.write(model_yaml)
		yaml_file.close()
		print("Saved config to: " + path_yaml)

		# serialize weights to HDF5
		model.save_weights(path_h5)
		print("Saved weight to: " + path_h5)

	@staticmethod
	def load_model(path):
		path_yaml = path + ".yaml"
		path_h5 = path + ".h5"

		# load config
		yaml_file = open(path_yaml, 'r')
		model_yaml = yaml_file.read()
		yaml_file.close()
		model = model_from_yaml(model_yaml)
		print("Loaded model from disk: " + path_yaml)
		
		# load weights into new model
		model.load_weights(path_h5)
		print("Loaded model from disk: " + path_h5)
		return (model)

	###########################################################################
	################################## IMAGE ##################################
	###########################################################################

	@staticmethod
	def image_to_array(path, convertColor):
		img = Image.open(path).convert(convertColor)
		arr = np.array(img)
		# make a 1-dimensional view of arr
		flat_arr = arr.ravel()
		return (flat_arr)

	@staticmethod
	def select_output(size, index):
		if (index < 0):
			raise Exception("Out of bound index at " + str(index))
		tmp = np.zeros(size)
		tmp[index] = 1
		return (tmp)

	@staticmethod
	def add_output_to_dataset(Y_train, size, index):
		if (Y_train is None):
			Y_train = KerasHelper.select_output(size, index)
		else:
			Y_train = np.vstack((Y_train, KerasHelper.select_output(size, index)))
		return (Y_train)

	@staticmethod
	def add_img_to_dataset(X_train, img):
		if (X_train is None):
			X_train = np.array(img)
		else:
			X_train = np.vstack((X_train, img))
		return (X_train)

	@staticmethod
	def get_dataset_with_folder(path, convertColor):
		X_train = None
		Y_train = None

		for foldername in os.listdir(path):
			print("Load folder: " + foldername)
			for filename in os.listdir(path + foldername):
				path2 = path + foldername + "/" + filename
				# img = KerasHelper.image_to_array_greyscale(path2)
				img = KerasHelper.image_to_array(path2, convertColor)
				X_train = KerasHelper.add_img_to_dataset(X_train, img)
				Y_train = KerasHelper.add_output_to_dataset(Y_train, 10, int(foldername))
		return (X_train, Y_train)

	@staticmethod
	def get_dataset_with_once_folder(name, path, convertColor):
		X_train = None
		Y_train = None

		# for foldername in os.listdir(path):
		# print "Load folder: " + path
		for filename in os.listdir(path):
			path2 = path + filename
			img = KerasHelper.image_to_array(path2, convertColor)
			X_train = KerasHelper.add_img_to_dataset(X_train, img)
			Y_train = KerasHelper.add_output_to_dataset(Y_train, 10, int(name))
		return (X_train, Y_train)

	###########################################################################
	################################### ELSE ##################################
	###########################################################################

	@staticmethod
	def log_level_decrease():
		os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Remove warning CPU SSE4 etc.

	@staticmethod
	def numpy_show_entire_array(px):
		lnbreak = (px + 1) * 4
		np.set_printoptions(threshold='nan', linewidth=lnbreak)

	@staticmethod
	def count_elem_in_folder(path):
		nb = 0
		for foldername in os.listdir(path):
			nb = nb + 1
		return (nb)
