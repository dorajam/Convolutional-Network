# Dora Jambor
# May 2016
# Use this to setup the convolutional network

from toy import *
from backprop import *

import collections

# import mnist_loader
# import ipdb; ipdb.set_trace()
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

ETA = 3
EPOCHS = 100
INPUT_SHAPE = (28*28)     # for mnist
HIDDEN_NEURONS = 30
OUTPUT_NEURONS = 10
BATCH_SIZE = 10
LMBDA = 0.1

'''
Args:
     Convolutional Layer: filter_size, stride, padding, num_filters
     Pooling Layer: poolsize
     Fully Connected Layer: num_output, classify = True/False, num_classes (if classify True)
     Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
'''
######################### TEST IMAGE ##########################
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

im = scipy.ndimage.imread('images/cat.jpg', flatten=True)
a = im.shape[0]
b= im.shape[1]
cat = scipy.misc.imresize(im, (a/40,b/40), interp='bilinear', mode=None)
# normalize
cat = 1.0 - cat/255.0

######################### TEST IMAGE ##########################

training_data = cat.reshape((1,43,64))
input_shape = training_data.shape
label = np.asarray(([1,0])).reshape((2,1))
training_data = (training_data, label)
# x,y = training_data[0][0].shape
# input_shape = (1,x,y)
print 'shape of input data: ', input_shape

net = Model(input_shape,
            layers = [
                {'conv_layer': {
                    'filter_size' : 3,
                    'stride' : 1,
                    'num_filters' : 3}},
                {'pool_layer': {
                    'poolsize' : (2,2)}},
                {'fc_layer1': {
                    'num_output' : 100,
                    'classify' : True,
                    'num_classes' : 2}}

            ])

net.gradient_descent(training_data, BATCH_SIZE, ETA, EPOCHS, LMBDA, test_data = None)



