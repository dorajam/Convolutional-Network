# Dora Jambor
# May 2016, run.py
# Use this to setup the convolutional network

import mnist_loader
from toy import *
from backprop import *

import collections


######################### TEST IMAGE ##########################
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# im = scipy.ndimage.imread('images/cat.jpg', flatten=True)
# a = im.shape[0]
# b= im.shape[1]
# cat = scipy.misc.imresize(im, (a/40,b/40), interp='bilinear', mode=None)
# # normalize
# cat = 1.0 - cat/255.0

######################### TEST IMAGE ##########################


ETA = 3
EPOCHS = 100
INPUT_SHAPE = (28*28)     # for mnist
BATCH_SIZE = 10
LMBDA = 0.1

# import ipdb; ipdb.set_trace()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


'''
Args:
     Convolutional Layer: filter_size, stride, padding, num_filters
     Pooling Layer: poolsize
     Fully Connected Layer: num_output, classify = True/False, num_classes (if classify True)
     Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
'''

# training_data = cat.reshape((1,43,64))
# input_shape = training_data.shape
# label = np.asarray(([1,0])).reshape((2,1))
# training_data = (training_data, label)
x,y = training_data[0][0].shape
input_shape = (1,x,y)
print 'shape of input data: ', input_shape

# net = Model(input_shape,
#             layers = [
#                 {'conv_layer': {
#                     'filter_size' : 5,
#                     'stride' : 1,
#                     'num_filters' : 20}},
#                 {'pool_layer': {
#                     'poolsize' : (2,2)}},
#                 {'fc_layer': {
#                     'num_output' : 100}},
#                 {'final_layer': {
#                     'num_classes' : 10}}
#             ])


net = Model(input_shape,
            layers = [
                {'fc_layer': {
                    'num_output' : 100}},
                {'final_layer': {
                    'num_classes' : 10}}
            ])

net.gradient_descent(training_data, BATCH_SIZE, ETA, EPOCHS, LMBDA, test_data = test_data[:10])
