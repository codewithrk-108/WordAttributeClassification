import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

import wandb
import random

import sys
# sys.path.append('../data')

from PIL import Image

from DataPreparation import *
import numpy as np

from imageio import imwrite
import os
import argparse
import sys
import random

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='DCGAN')

parser.add_argument('--img-dir', type=str, default='./data/celebA',
					help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='./output',
					help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
					help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
					help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--z-dim', type=int, default=100,
					help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=128,
					help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=2,
					help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=50,
					help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.01,
					help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
					help='"beta1" parameter for Adam optimizer')

parser.add_argument('--num-gen-updates', type=int, default=2,
					help='Number of generator updates per discriminator update')

parser.add_argument('--log-every', type=int, default=5,
					help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
					help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
					help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()





class AutoEncoder(tf.keras.Model):
	def __init__(self,lr,batch_size,epoch):
		super(AutoEncoder, self).__init__()

		self.learning_rate = lr
		self.optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate,momentum=0.9,decay=0.0005,nesterov=True)
		self.batch_size = batch_size
		self.epoch = epoch

		wandb.init(
    	# set the wandb project where this run will be logged
    	project="Autoencoder",
    
    	# track hyperparameters and run metadata
    	config={
    	"learning_rate": self.learning_rate,
    	"architecture": "Autoencoder",
    	"dataset": "CustomDataset",
    	"epochs": self.epoch,
    		}
		)
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

		self.stride_size = 2
		self.reshape = tf.keras.layers.Reshape((50, 50, 1))
		self.conv_layer1 = Conv2D(input_shape=(50,50,1), filters=64, strides=1, kernel_size=(3,3),activation=self.leaky_relu, name='conv_layer1', dtype=tf.float32)
		self.pooling = MaxPooling2D(pool_size=(2, 2))
		self.conv_layer2 = Conv2D(filters=128, strides=1, kernel_size=(2,2), activation=self.leaky_relu, padding='same', name='conv_layer2', dtype=tf.float32)
		self.deconv_layer1 = Conv2DTranspose(filters=64, strides=1, kernel_size=(2,2), activation=self.leaky_relu, padding='same', name='deconv_layer1', dtype=tf.float32)
		self.unpooling = UpSampling2D(size=(2, 2))
		self.deconv_layer2 = Conv2DTranspose(filters=1, strides=1, kernel_size=(3,3), activation=self.leaky_relu,name='deconv_layer2', dtype=tf.float32)


	def call(self, inputs):
		""" Input: training inputs
			Output: result of inputs being passed through the autoencoder.
		"""
		inputs = self.reshape(inputs)
		c1 = self.conv_layer1(inputs)
		pool = self.pooling(c1)
		c2 = self.conv_layer2(pool)
		d1 = self.deconv_layer1(c2)
		upsample = self.unpooling(d1)
		d2 = self.deconv_layer2(upsample)
		return d2

	def loss(self, original, decoded):
		""" Input: original - the raw cropped images
				   decoded - the decoded images passed through the autoencoder
			Output: Returns the loss (calculated with MSE) between original and decoded images.
		"""
		mse = tf.keras.losses.MeanSquaredError()
		decoded = tf.squeeze(decoded)
		add=0
		for img in (decoded-original):
			# print(np.count_nonzero(img==0))
			if(np.sum(img==0)==img.shape[0]*img.shape[1]):
				add+=1
		return add,mse(original, decoded)

def val_test(X_val,result):
	result = tf.squeeze(result)

	mse_val = tf.keras.losses.MeanSquaredError()
	return mse_val(X_val,result)


def train(model, real_images, fake_images):
	""" Input: real_images - real unlabeled data
			   fake_images - synthetic data (labels withheld)
		Output: None

		Trains the autoencoder using a shuffled mix of real and synthetic images.
	"""


	X_train_real,X_val_real = train_test_split(real_images,train_size=0.80,shuffle=True,random_state=42)
	X_train_syn,X_val_syn = train_test_split(fake_images,train_size=0.80,shuffle=True,random_state=42)

	x_val = np.concatenate((X_val_real,X_val_syn),axis=0)

	del real_images
	del fake_images
	gc.collect()

	del X_val_syn
	del X_val_real
	gc.collect()

	iterations = (len(X_train_real) + len(X_train_syn)) // model.batch_size
	fake_batch = len(X_train_syn) // iterations
	real_batch = len(X_train_real) // iterations

	total_loss = 0
	acc=0
	acc_val=0
	loss_val=0
	val_batch=300
	iter = len(x_val)//val_batch

	itr=0

	for i in range(iterations):

		real_inputs = X_train_real[i * real_batch : (i+1) * real_batch]
		fake_inputs = X_train_syn[i * fake_batch : (i+1) * fake_batch]

		inputs = np.concatenate((real_inputs, fake_inputs), axis=0)

		with tf.GradientTape() as tape:
			res = model(inputs,training=True)
			add,loss = model.loss(inputs, res)
		total_loss += loss
		acc+=add

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	

	while(itr<len(x_val)):
		result = model.call(x_val[itr:itr+val_batch])
		loss_val += val_test(x_val[itr:itr+val_batch],result)
		itr+=val_batch
	total_loss = total_loss / float(iterations)
	wandb.log({"train_loss":total_loss,"val_loss":loss_val/float(iterations)})

def test(model, real_images, synthetic_images):
	""" Input: original - the raw cropped images
			   decoded - the decoded images passed through the autoencoder
		Output: Returns the loss (calculated with MSE) between original and decoded images.
	"""
	# 1) Call the encoder, and decoder outputs
	# 2) Save encoder + decoder on some samples.
	real_inputs = real_images[:40] #num is adjustable.

	count = 0
	for real in real_inputs:
		for row in range(len(real)):
			for col in range(len(real[0])):
				real[row][col] = int(real[row][col] * 255)

		real = np.array(real, dtype=np.uint8)
		real = Image.fromarray(real)
		real = real.convert('L')
		real.save("./scae_in/"+ str(count) + ".png", format='PNG')
		count += 1

	res = model.call(real_inputs)
	res = np.array(res)
	res = np.squeeze(res)
	res_count = 0
	for real in res:
		real /= np.max(real)/255.0
		real = np.clip(real, 0, 255)
		for row in range(len(real)):
			for col in range(len(real[0])):
				real[row][col] = int(real[row][col])
		real = np.array(real, dtype=np.uint8)
		real = Image.fromarray(real)
		real = real.convert('L')
		real.save("./scae_out/"+ str(res_count) + ".png", format='PNG')
		res_count += 1

def main():
	autoencoder = AutoEncoder(0.01,128,50)

	# For saving/loading models
	checkpoint_dir = './checkpoints_ae'
	# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(model=autoencoder)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10,step_counter=5)
	# Ensure the output directory exists
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	if args.restore_checkpoint or args.mode == 'test':
		# restores the latest checkpoint using from the manager
		checkpoint.restore(manager.latest_checkpoint)
		
	
	try:
		# Specify an invalid GPU device
		# print(autoencoder.summary())
		with tf.device('/device:' + args.device):
			if args.mode == 'train':
				real_images, synthetic_images = get_data_for_autoencoder("/scratch/sahi_data/ae_real_inputs.hdf5", "/scratch/sahi_data/synthetic_scae_inputs.hdf5")
				print(len(real_images),len(synthetic_images[:45000]))
				for im in real_images:
					im = np.clip(im,0,1)
				for im in synthetic_images:
					im = np.clip(im,0,1)

				for epoch in range(0, args.num_epochs):
					print('========================== EPOCH %d  ==========================' % epoch)
					train(autoencoder, real_images, synthetic_images[:45000])
					print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
					manager.save()
					autoencoder.save_weights('./weights/weights_leaky_relu.h5')
			if args.mode == 'test':
				real_images, synthetic_images = get_data_for_autoencoder("/scratch/ae_real_inputs.hdf5", "/scratch/synthetic_scae_inputs.hdf5")
				test(autoencoder, real_images, synthetic_images)
		wandb.finish()
	except RuntimeError as e:
		print(e)

if __name__== "__main__":
	main()
