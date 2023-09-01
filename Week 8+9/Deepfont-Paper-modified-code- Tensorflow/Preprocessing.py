import os
from scipy import misc
import numpy as np
from PIL import Image, ImageFile
import random
import json
import pickle
import cv2
import h5py
import tensorflow as tf
import gc

import re
import matplotlib.pyplot as plt
import matplotlib as mpl
# from create_train_set_labels import font_dict

# print(font_dict)


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------- IMAGE PREPROCESSING --------------------------------------#
def resize_image(image, image_dimension):
	""" Input: Image Path
		Output: Image
		Resizes image to height of 105px, while maintaining aspect ratio
	"""
	base_height = image_dimension
	img = image
	height_percent = (base_height/float(img.size[1]))
	wsize = int((float(img.size[0])*float(height_percent)))
	# print("Width", wsize)
	img = img.resize((wsize, base_height),Image.ANTIALIAS )

	return img

# img , img_dimension = 105, 
def generate_crop(img, image_dimension, num_vals):
	""" Input: Image object, the width and height of our image, number of cropped images
		Output: A list of mnormalized numpy arrays normalized between 0 and 1.
		Randomly generates 15 cropped images.
	"""
	cropped_images = []
	width = len(np.array(img)[1])

	if width > image_dimension + num_vals:
		bounds = random.sample(range(0, width-image_dimension), num_vals)
		for i in range(num_vals):
			# left top right bottom
			new_img = img.crop((bounds[i], 0, bounds[i] + image_dimension, image_dimension))
			new_img = np.array(new_img) / 255.0

			cropped_images.append(new_img)
	return cropped_images

def generate_crop_samples(root_dir):
	""" Input: root_dir, directory of desired images
		Output: none
		Creates some cropping samples that can be viewed.
	"""
	count = 0

	for subdir in os.listdir(root_dir): 
		subdir_path = root_dir + "/" + subdir
		font_name = subdir

		for file in os.listdir(subdir_path):
			if count == 3:
				break
			image_path = subdir_path + "/" + file
			image = alter_image(image_path)
			image = resize_image(image, 96)
			cropped_images = generate_crop(image, 96, 10)
			count+=1

	crop_count = 0
	for crop in cropped_images:
		for row in range(len(crop)):
			for col in range(len(crop[0])):
				crop[row][col] = int(crop[row][col] * 255)

		crop = np.array(crop, dtype=np.uint8)
		crop = Image.fromarray(crop)
		crop = crop.convert('L')
		crop.save("./crops/"+ str(crop_count) + ".png", format='PNG')
		crop_count += 1

def alter_image(image_path):
	""" Input: Image path
		Output: Altered image object
		Function to apply all of the filters (noise, blur, perspective rotation & translation) to a single image.
	"""

	# print(image_path)
	img = Image.open(image_path)
	img = img.convert("L") #convert image to grey scale]
	img = np.array(img)

	# noise
	row, col = img.shape
	gauss = np.random.normal(0, 3, (row, col))
	gauss = gauss.reshape(row, col)
	noised_image = img + gauss

	# blur
	blurred_image = cv2.GaussianBlur(noised_image, ksize = (3, 3), sigmaX = random.uniform(2.5, 3.5))

	# perspective transform and translation
	rotation_angle = [-4, -2, 0, 2, 4]
	translate_x = [-5, -3, 0, 3, 5]
	translate_y = [-5, -3, 0, 3, 5]
	angle = random.choice(rotation_angle)
	angle = random.choice(rotation_angle)
	angle = random.choice(rotation_angle)
	tx = random.choice(translate_x)
	ty = random.choice(translate_y)

	rows, cols = img.shape
	M_translate = np.float32([[1,0,tx],[0,1,ty]])
	M_rotate = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

	affined_image = cv2.warpAffine(blurred_image, M_translate, (cols, rows))
	affined_image = cv2.warpAffine(affined_image, M_rotate, (cols, rows))

	# shading
	# changed to laplacian
	# print(affined_image)
	# print("----------")
	affined_image = np.array(affined_image) * random.uniform(0.2, 1.5)
	# affined_image = cv2.Laplacian(affined_image,cv2.CV_64F)
	# affined_image = cv2.convertScaleAbs(affined_image)
	affined_image = np.clip(affined_image, 0, 255).astype(np.uint8)
	final_image = Image.fromarray(affined_image)

	return final_image



# ----------------------------- SPLITTING SYNTHETIC DATA --------------------------------------#
def create_hdf5(root_dir):
	""" Input: Root directory (string)
		Output: Creates hdf5 files to use for our model.

		Processes synthetic data by segmenting them into the following
		# 1) Synthetic train inputs for autoencoder - 10%
		2) Train input & labels for DeepFont model - 100%
		# 3) Test input & labels for DeepFont Model - 10%
	"""

	train_inputs = np.zeros((4*2383*6,50,50))
	train_labels = np.zeros((train_inputs.shape[0],4))
	test_inputs = np.zeros((2*2383*6,50,50))
	test_labels = np.zeros((test_inputs.shape[0],4))
	scae_inputs = np.zeros((3*2383*6,50,50))

	# 4 + 2 + 3 = 9

	with open('backwards_font_dict.json') as json_file:
		font_subset = json.load(json_file)

	gc.collect()
	for font_folder in os.listdir(root_dir):
		fol_path = os.path.join(root_dir,font_folder) 
		array_img = np.random.choice(os.listdir(fol_path),9)
		
		for itr,img in enumerate(array_img[:4]): # goes through all font folders
			# print(img)
			image_path = fol_path + "/" + img
			# print(image_path)
			lbl = font_dict[98]
			lbl = font_dict[int(font_folder)]
			# lbl = list(map(int,lbl))
			image = alter_image(image_path)

			image = resize_image(image, 50)
			cropped_images = generate_crop(image, 50, 6)
		
			for c in cropped_images:
				train_inputs[itr,:] = c
				# train_labels[itr] = lbl
				for el in lbl:
					train_labels[itr][el]=1
	
		for itr,img in enumerate(array_img[4:7]): # goes through all font folders
			image_path = fol_path + "/" + img
			lbl = font_dict[int(font_folder)]
			# lbl = image_path.split('.')[-2].split(',')
			# lbl = list(map(int,lbl))
			image = alter_image(image_path)

			image = resize_image(image, 50)
			cropped_images = generate_crop(image, 50, 6)
		
			for c in cropped_images:
				scae_inputs[itr,:] = c

		for itr,img in enumerate(array_img[7:9]): # goes through all font folders
			image_path = fol_path + "/" + img
			lbl = font_dict[int(font_folder)]
			image = alter_image(image_path)

			image = resize_image(image, 50)
			cropped_images = generate_crop(image, 50, 6)
		
			for c in cropped_images:
				test_inputs[itr,:] = c
				for el in lbl:
					test_inputs[itr][el]=1


	shuffle_and_save_autoencoder(scae_inputs, "synthetic_scae_inputs")
	shuffle_and_save(train_inputs, "train_inputs", train_labels, "train_labels")
	shuffle_and_save(test_inputs, "test_inputs", test_labels, "test_labels")

	print("Finished preprocessing...")



# ----------------------------- AUTOENCODER SPECIFIC FUNCS --------------------------------------#
def process_unlabeled_real(root_dir):
	""" Input: Root directory (string) - Train inputs for Autoencoder

		Preprocess the unlabeled real data.
	"""
	ae_real_inputs = []

	print("Starting processing of unlabeled real...")
	count = 0

	# check not include all images
	# ! must check
	cnt=0
	for f in os.scandir(root_dir):
		# print(cnt)
		if (cnt>=45000):
			break
		if (f.name.endswith(".jpeg") or f.name.endswith(".jpg") or f.name.endswith(".png")):

			image_path = f.path

			image = alter_image(image_path)
			image = resize_image(image, 50)

			if count % 13000 == 0:
				count_str = str(count)
				print( "Images preprocessed: ", count)

			cropped_images = generate_crop(image, 50, 6)

			for c in cropped_images:
				ae_real_inputs.append(c)
			cnt+=6

		count += 1

	print("Number of images in file: ", len(ae_real_inputs))
	with h5py.File('ae_real_inputs.hdf5', 'w') as f:
		 f.create_dataset('ae_real_inputs',data=ae_real_inputs)


def get_data_for_autoencoder(real_filename, synthetic_filename):
	""" input: file paths to the desires files
	"""
	with h5py.File(synthetic_filename, 'r') as hf:
		# print(synthetic_filename[0:-5])
		name = synthetic_filename.split('/')[-1].split('.')[-2]
		ae_synthetic_inputs = hf[name][:]
	with h5py.File(real_filename, 'r') as hf:
		name = real_filename.split('/')[-1].split('.')[-2]
		ae_real_inputs = hf[name][:]

	return ae_real_inputs, ae_synthetic_inputs




# ----------------------------- DEEPFONT SPECIFIC FUNCS --------------------------------------#
def get_train_df(inputs_filename, labels_filename):
	""" Input: None
		Output: None

		Opens the train inputs and train labels and returns them.
	"""
	with h5py.File(inputs_filename, 'r') as hf:
		train_inputs = hf[inputs_filename[0:-5]][:]

	print("Finished opening train inputs")

	with h5py.File(labels_filename, 'r') as hf:
		train_labels = hf[labels_filename[0:-5]][:]

	print("Finished opening train labels")

	return train_inputs, train_labels

def get_test_df(inputs_filename, labels_filename):
	""" Input: None
		Output: None

		Opens the test inputs and test labels and returns them.
	"""
	with h5py.File(labels_filename, 'r') as hf:
		test_labels = hf[labels_filename[0:-5]][:]

	print("Finished opening test labels")

	with h5py.File(inputs_filename, 'r') as hf:
		test_inputs = hf[inputs_filename[0:-5]][:]

	print("Finished opening test inputs")

	return test_inputs, test_labels




# ----------------------------- SHUFFLING FUNCTIONS --------------------------------------#
def train_shuffle():
	""" Input: None
		Output: None

		Shuffles the training set in groups of 10.
	"""
	print("Shuffling...")

	shuffle_size = 10

	with h5py.File('train_inputs.hdf5', 'r') as hf:
		train_inputs = hf['train_inputs'][:]

	print("train labels finished")

	with h5py.File('train_labels.hdf5', 'r') as hf:
		train_labels = hf['train_labels'][:]

	temp = list(range(len(train_inputs)//shuffle_size)) # list with all the indices of test_inputs divided by ten?
	random.shuffle(temp) #
	train_inputs_copy = train_inputs[:]
	train_labels_copy = train_labels[:]

	for i, j in enumerate(temp):
		if not i == j:
			train_inputs_copy[i*shuffle_size:(i+1)*shuffle_size] = train_inputs[j*shuffle_size:(j+1)*shuffle_size]
			train_labels_copy[i*shuffle_size:(i+1)*shuffle_size] = train_labels[j*shuffle_size:(j+1)*shuffle_size]

	train_inputs_copy = np.array(train_inputs_copy)
	train_labels_copy = np.array(train_labels_copy)

	with h5py.File('shuffled_train_inputs.hdf5', 'w') as f:
		f.create_dataset("shuffled_train_inputs",data=train_inputs_copy)

	with h5py.File('shuffled_train_labels.hdf5', 'w') as f:
		f.create_dataset("shuffled_train_labels",data=train_labels_copy)

	print("done shuffling!")

def shuffle_and_save(inputs, inputs_file_name, labels, labels_file_name):
	""" Input: inputs, desired inputs file name, labels, desired labels file name, group number to shuffle by
		Output: None

		Shuffles the inputs and labels by a shuffle_size and then dumps them into hdf5 files.
	"""
	# test_inputs = inputs
	# test_labels = labels

	# temp = list(range(len(test_inputs)//shuffle_size)) # list with all the indices of test_inputs divided by ten?
	# random.shuffle(temp) #
	# test_inputs_copy = test_inputs[:]
	# test_labels_copy = test_labels[:]

	# for i, j in enumerate(temp):
	# 	if not i == j:
	# 		test_inputs_copy[i*shuffle_size:(i+1)*shuffle_size] = test_inputs[j*shuffle_size:(j+1)*shuffle_size]
	# 		test_labels_copy[i*shuffle_size:(i+1)*shuffle_size] = test_labels[j*shuffle_size:(j+1)*shuffle_size]

	# test_inputs_copy = np.array(test_inputs_copy)
	# test_labels_copy = np.array(test_labels_copy)

	with h5py.File(inputs_file_name + '.hdf5', 'w') as f:
		f.create_dataset(inputs_file_name,data=inputs)

	with h5py.File(labels_file_name + '.hdf5', 'w') as f:
		f.create_dataset(labels_file_name,data=labels)

def shuffle_and_save_autoencoder(inputs, inputs_file_name):
	""" Input: synthetic inputs, filename to save them under
		Output: none

		Shuffles and saves the synthetic inputs for the autoencoder.
	"""

	with h5py.File(inputs_file_name + '.hdf5', 'w') as f:
		 f.create_dataset(inputs_file_name,data=inputs)



# ----------------------------- TESTING DATA FOR DEEPFONT --------------------------------------#
def get_real_test(root_dir):
	""" Input: directory of real test data
		Output: None

		Preprocesses the real test data and returns real test inputs, real test labels,
	"""
	with open('150_fonts.json') as json_file:
		font_subset = json.load(json_file)

	real_test_inputs = []
	real_test_labels = []

	total_folder_count  = 0
	# for subdir in os.listdir(root_dir):
		# if subdir in font_subset:
		# 	subdir_path = root_dir + "/" + subdir
		# 	font_name = subdir

		# 	file_count = 0
		# 	for file in os.listdir(subdir_path): # goes through all sample images
		# 		image_path = subdir_path + "/" + file
		# 		image = alter_image(image_path)
		# 		image = resize_image(image, 96)
		# 		cropped_images = generate_crop(image, 96, 10)

		# 		for c in cropped_images:
		# 			real_test_inputs.append(c)
		# 			real_test_labels.append(font_subset[font_name])

		# 		file_count += 1

		# if total_folder_count % 100 == 0:
		# 	print(total_folder_count, "folders done")
		# total_folder_count += 1

	return real_test_inputs, real_test_labels

def combine_real_synthetic_test():
	""" Input: None
		Output: None

		Combines the real testing data with the synthetic testing data, shuffles, then saves them as hdf5 files.
	"""
	real_inputs, real_labels = get_real_test("./VFR_real_test")
	print("finished processing real inputs & labels")

	with h5py.File('test_labels.hdf5', 'r') as hf:
		synth_labels = hf['test_labels'][:]

	with h5py.File('test_inputs.hdf5', 'r') as hf:
		synth_inputs = hf['test_inputs'][:]

	combined_inputs = np.concatenate((synth_inputs, real_inputs), axis=0)
	combined_labels = np.concatenate((synth_labels, real_labels), axis=0)
	shuffle_and_save(combined_inputs, "combined_test_inputs", combined_labels, "combined_test_labels")
	print("finished shuffling")

def check_labels_and_inputs():
	""" Input: None
		Output: None

		Function to check labels and inputs.
	"""
	with h5py.File('combined_test_labels.hdf5', 'r') as hf:
		combined_labels = hf['combined_test_labels'][:]

	with h5py.File('combined_test_inputs.hdf5', 'r') as hf:
		combined_inputs = hf['combined_test_inputs'][:]

	print("CHECKING... COMBINED LABELS", combined_labels[0:100])
	print("CHECKING... COMBINED INPUTS", combined_inputs[0:100])




# ----------------------------- DICTIONARY FUNCTIONS --------------------------------------#

# No use
def create_total_font_dictionary():
	""" Input: none
		Output: none
		Creates a font dictionary based on the entire font library used by the authors of DeepFont.
		dict key is fontname, dict val is index
	"""
	path = "./fontlist.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 1
	for line in content:
		dict[line] = count
		count += 1
	with open('font_dict.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)

# No use
def create_total_font_dictionary_backwards():
	""" Input: none
		Output: none
		Creates a font dictionary based on the entire font library used by the authors of DeepFont.
		dict key is index, dict val is fontname
	"""
	path = "./fontlist.txt"

	f = open(path, 'r')
	content = f.read().split()
	dict = {}
	count = 1
	for line in content:
		dict[count] = line
		count += 1
	with open('backwards_font_dict.json', 'w') as fp:
		json.dump(dict, fp,  indent=4)


# Gets the dictionary font : index
def get_font_dict():
	""" Input: none
		Output: font dict
		Opens font dict and returns it.
	"""
	with open('font_dict.json') as json_file:
		font_dict = json.load(json_file)
	return font_dict

# ----------------------------- MAIN ----------------------------------#
def main():
	print ("We used main to run our preprocess functions. :]")
	# process_unlabeled_real("./scrape-wtf-new")
	# print(get_font_dict())
	# img = resize_image(Image.open('./13.png'),50)
	# img = np.array(img)
	# print(img.shape)
	# plt.imshow(img,cmap='gray')
	# plt.show()
	# print(np.unique(img))
	# cv2.imshow('new_image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# create_hdf5("/home/rohan/Downloads/Adobe_VFR/VFR_Dataset/AD_OBE/train")
	# process_unlabeled_real("/home/rohan/Downloads/Adobe_VFR/VFR_Dataset/AD_OBE/real_u")
	
	# img = alter_image("/home/rohan/Downloads/Adobe_VFR/VFR_CLassification_demo/Font_Recognition-DeepFont/WhatTheFont-master/data/real_test_sample/ACaslonPro-Bold/ACaslonPro-Bold1276.png")
	# print(img.shape)
	# img.show('cool')
	# img = np.array(img)

	# cv2.imshow('new',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# generate_crop_samples("./real_test_sample")


if __name__ == "__main__":
	main()
