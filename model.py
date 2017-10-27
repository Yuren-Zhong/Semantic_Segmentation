
from keras.layers import Activation, Reshape, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras.models import Sequential

import time

def add_softmax(model: Sequential) -> Sequential:
	_, curr_width, curr_height, curr_channels = model.layers[-1].output_shape

	model.add(Reshape((curr_width * curr_height, curr_channels)))
	model.add(Activation('softmax'))

	return model

def dilated_frontend(input_width, input_height) -> Sequential:
	model = Sequential()

	model.add(Conv2D(64, (3,3), activation='relu', input_shape=(input_width, input_height, 3)))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(128, (3,3), activation='relu'))
	model.add(Conv2D(128, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(512, (3,3), activation='relu'))
	model.add(Conv2D(512, (3,3), activation='relu'))
	model.add(Conv2D(512, (3,3), activation='relu'))

	model.add(Conv2D(512, (3,3), dilation_rate=(2,2), activation='relu'))
	model.add(Conv2D(512, (3,3), dilation_rate=(2,2), activation='relu'))
	model.add(Conv2D(512, (3,3), dilation_rate=(2,2), activation='relu'))

	# dilated
	model.add(Conv2D(4096, (7,7), dilation_rate=(4,4), activation='relu'))
	model.add(Dropout(0.5, seed=int(time.time())))
	model.add(Conv2D(4096, (1,1), activation='relu'))
	model.add(Dropout(0.5, seed=int(time.time()) + int(time.time())))
	model.add(Conv2D(21, (1,1), activation='linear'))

	return model

def deconv_frontend(input_width, input_height) -> Sequential:
	model = Sequential()

	model.add(Conv2D(64, (3,3), activation='relu', input_shape=(input_width, input_height, 3)))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(128, (3,3), activation='relu'))
	model.add(Conv2D(128, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(512, (3,3), activation='relu'))
	model.add(Conv2D(512, (3,3), activation='relu'))
	model.add(Conv2D(512, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))

	# fully conv
	model.add(Conv2D(4096, (7,7), activation='relu'))
	model.add(Dropout(0.5, seed=int(time.time())))
	model.add(Conv2D(4096, (1,1), activation='relu'))
	model.add(Dropout(0.5, seed=int(time.time()) + int(time.time())))
	model.add(Conv2D(21, (1,1)))
	model.add(Conv2DTranspose(21, (64,64), strides=32))



	return model