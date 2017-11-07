
import argparse
import os

import numpy as np
from PIL import Image
from IPython import embed

from model import dilated_frontend, add_softmax
from utils import interp_map, pascal_palette

input_width, input_height = 900, 900
label_margin = 186

has_context_module = False

input_path = 'images/cat.jpg'
output_path = None
mean = [102.93, 111.36, 116.52]
zoom = 8
weights_path = 'trained_log/2017-10-27 17:29-lr1e-04-bs001/ep42-vl1.0240.hdf5'
modeltype = 'dilated'

def get_trained_model():
	""" Returns a model with loaded weights. """
	global weights_path

	if modeltype == 'dilated':
		model = dilated_frontend(input_width, input_height)
	else:
		pass

	model = add_softmax(model)

	def load_tf_weights():
		""" Load pretrained weights converted from Caffe to TF. """

		# 'latin1' enables loading .npy files created with python2
		weights_data = np.load(weights_path, encoding='latin1').item()

		for layer in model.layers:
			if layer.name in weights_data.keys():
				layer_weights = weights_data[layer.name]
				layer.set_weights((layer_weights['weights'], layer_weights['biases']))

	def load_keras_weights():
		""" Load a Keras checkpoint. """
		model.load_weights(weights_path)

	if weights_path.endswith('.npy'):
		load_tf_weights()
	elif weights_path.endswith('.hdf5'):
		load_keras_weights()
	else:
		raise Exception("Unknown weights format.")

	return model

def forward_pass():
	global input_path
	global output_path
	global mean
	global zoom
	global weights_path

	if not output_path:
		dir_name, file_name = os.path.split(input_path)
		output_path = os.path.join(
			dir_name,
			'{}_seg.png'.format(
			os.path.splitext(file_name)[0]))

	model = get_trained_model()

	# Load image and swap RGB -> BGR to match the trained weights
	image_rgb = np.array(Image.open(input_path)).astype(np.float32)
	image = image_rgb[:, :, ::-1] - mean
	image_size = image.shape

	# Network input shape (batch_size=1)
	net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

	output_height = input_height - 2 * label_margin
	output_width = input_width - 2 * label_margin

	# This simplified prediction code is correct only if the output
	# size is large enough to cover the input without tiling
	assert image_size[0] < output_height
	assert image_size[1] < output_width

	# Center pad the original image by label_margin.
	# This initial pad adds the context required for the prediction
	# according to the preprocessing during training.
	image = np.pad(image,
			((label_margin, label_margin),
			(label_margin, label_margin),
			(0, 0)), 'reflect')

	# Add the remaining margin to fill the network input width. This
	# time the image is aligned to the upper left corner though.
	margins_h = (0, input_height - image.shape[0])
	margins_w = (0, input_width - image.shape[1])
	image = np.pad(image, (margins_h, margins_w, (0, 0)), 'reflect')

	# Run inference
	net_in[0] = image
	prob = model.predict(net_in)[0]

	# Reshape to 2d here since the networks outputs a flat array per channel
	prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
	prob = prob.reshape((prob_edge, prob_edge, 21))

	# Upsample
	if zoom > 1:
		prob = interp_map(prob, zoom, image_size[1], image_size[0])

	# Recover the most likely prediction (actual segment class)
	prediction = np.argmax(prob, axis=2)

	# Apply the color palette to the segmented image
	color_image = np.array(pascal_palette)[prediction.ravel()].reshape(
		prediction.shape + (3,))

	print('Saving results to: ', output_path)
	with open(output_path, 'wb') as out_file:
		Image.fromarray(color_image).save(out_file)


if __name__ == "__main__":
	forward_pass()
