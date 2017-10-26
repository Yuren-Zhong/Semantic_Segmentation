#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil

import numpy as np
from keras import callbacks, optimizers
from IPython import embed

from model import deconv_frontend, dilated_frontend, add_softmax
from image_reader import RandomTransformer, SegmentationDataGenerator

train_list_fname = '/home/v-yurzho/benchmark_RELEASE/dataset/train.txt'
val_list_fname = '/home/v-yurzho/benchmark_RELEASE/dataset/val.txt'
img_root = '/home/v-yurzho/benchmark_RELEASE/dataset/img'
mask_root = '/home/v-yurzho/benchmark_RELEASE/dataset/pngs'
weights_path = '/home/v-yurzho/conversion/converted/dilation8_pascal_voc.npy'
batch_size = 1
learning_rate = 1e-4

def load_weights(model, weights_path):
	print("load weights")
	weights_data = np.load(weights_path, encoding='latin1').item()

	for layer in model.layers:
		if layer.name in weights_data.keys():
			layer_weights = weights_data[layer.name]
			layer.set_weights((layer_weights['weights'],
				layer_weights['biases']))

def build_abs_paths(basenames):
	global img_root
	global mask_root
	img_fnames = [os.path.join(img_root, f) + '.jpg' for f in basenames]
	mask_fnames = [os.path.join(mask_root, f) + '.png' for f in basenames]
	return img_fnames, mask_fnames

def train():
	global train_list_fname
	global val_list_fname
	global img_root
	global mask_root
	global weights_path
	global batch_size
	global learning_rate

	train_data_gen = SegmentationDataGenerator(
		RandomTransformer(horizontal_flip = True, vertical_flip = True))
	val_data_gen = SegmentationDataGenerator(
		RandomTransformer(horizontal_flip = True, vertical_flip = True))

	trained_log = '{}-lr{:.0e}-bs{:03d}'.format(
		time.strftime("%Y-%m-%d %H:%M"),
		learning_rate,
		batch_size)
	checkpoints_folder = 'trained_log/' + trained_log
	try:
		os.makedirs(checkpoints_folder)
	except OSError:
		shutil.rmtree(checkpoints_folder, ignore_errors=True)
		os.makedirs(checkpoints_folder)

	model_checkpoint = callbacks.ModelCheckpoint(
		checkpoints_folder + '/ep{epoch:02d}-vl{val_loss:.4f}.hdf5', monitor='loss')
	model_tensorboard = callbacks.TensorBoard(
		log_dir='{}/tboard'.format(checkpoints_folder),
		histogram_freq=0,
		write_graph=False,
		write_images=False)
	model_csvlogger = callbacks.CSVLogger(
		'{}/history.log'.format(checkpoints_folder))
	model_reducelr = callbacks.ReduceLROnPlateau(
		monitor='val_loss',
		factor=0.2,
		patience=5,
		verbose=1,
		min_lr=0.05 * learning_rate)

	model = add_softmax(dilated_frontend(500, 500))

	#load_weights(model, weights_path)

	model.compile(
		optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

	train_basenames = [l.strip() for l in open(train_list_fname).readlines()]
	val_basenames = [l.strip() for l in open(val_list_fname).readlines()][:500]

	train_img_fnames, train_mask_fnames = build_abs_paths(train_basenames)
	val_img_fnames, val_mask_fnames = build_abs_paths(val_basenames)

	model_skipped = callbacks.LambdaCallback(
		on_epoch_end=lambda a, b: open(
			'{}/skipped.txt'.format(checkpoints_folder), 'a').write(
			'{}\n'.format(train_data_gen.skipped_count)))

	model.fit_generator(
		train_data_gen.flow_from_list(
			train_img_fnames,
			train_mask_fnames,
			shuffle=True,
			batch_size=batch_size,
			img_target_size=(500,500),
			mask_target_size=(16, 16)),
		steps_per_epoch=(len(train_basenames)/batch_size),
		epochs=20,
		validation_data=val_data_gen.flow_from_list(
			val_img_fnames,
			val_mask_fnames,
			batch_size=8,
			img_target_size=(500,500),
			mask_target_size=(16,16)),
		validation_steps=(len(val_basenames)/8),
		callbacks=[
			model_checkpoint,
			model_tensorboard,
			model_csvlogger,
			model_reducelr,
			model_skipped
		])


if __name__ == '__main__':
	train()
