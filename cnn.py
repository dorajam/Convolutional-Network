# This is the basic idea behind the architecture

import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

from backprop import *

'''
im = scipy.ndimage.imread('cat.jpg', flatten=True)
print im.shape, type(im)
a = im.shape[0]
b= im.shape[1]
cat = scipy.misc.imresize(im, (a/40,b/40), interp='bilinear', mode=None)

# normalize
cat = 1.0 - cat/255.0
# print cat[30:40]
'''
# arr = np.zeros((cat2.shape[0], cat2.shape[1]))
# block_size = 4
# res = 0
# i= 0
# j = 0
# sub = 0
# k = 0
# while k < 10:
#     while i < cat2.shape[0] * cat2.shape[1]:
#         while j < 10:
#              sub += cat2[k][j]
#              j += 1
#              i += 1
#     arr.append(sub/block_size)
#     k += 2

# plt.imshow(cat2, cmap=plt.cm.gray, vmin=30, vmax=255)
# # plt.show()

# INPUT NEURONS = image
# cat = temp.reshape((temp.shape[0] * temp.shape[1], 1))


''' RECEPTIVE FIELD - WEIGHTS aka FILTER ->
initialize filters in a way that corresponds to the depth of the image.
If the input image is of channel 3 (RGB) then each of your weight vector is n*n*3.
PARAMETERS you'll need: NUM_FILTERS (num of filters), STRIDE (slide filter by), ZERO-PADDING(to control the spatial size of the output volumes). Use (Inputs-FilterSize + 2*Padding)/Stride + 1 to calculate your output volume and to decide your hyperparameters'''

DEPTH = 3
NUM_FILTERS = 3
STRIDE = 1
# to ensure that the input and output volumes are the same: use P=(F-1)/2 given stride 1.
PADDING = 0
FILTER_SIZE = 2

class ConvLayer(object):

    def __init__(self, input_shape, filter_size, stride, num_filters, padding = 0):
        self.depth, self.height_in, self.width_in = input_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters

        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size)
        self.biases = np.random.rand(self.num_filters,1)

        self.output_dim1 = (self.height_in - self.filter_size + 2*self.padding)/self.stride + 1        # num of rows
        self.output_dim2 =  (self.width_in - self.filter_size + 2*self.padding)/self.stride + 1         # num of cols
        
        
        self.z_values = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))

        print 'shape of input (depth, rows,cols): ', input_shape
        print 'shape of convolutional layer (depth, rows, cols): ', self.output.shape


    def convolve(self, input_neurons):
        '''
        Pass in the actual input data and do the convolution.
        Returns: sigmoid activation matrix after convolution 
        '''

        # roll out activations
        self.z_values = self.z_values.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        
        act_length1d =  self.output.shape[1]

        for j in range(self.num_filters):
            slide = 0
            row = 0

            for i in range(act_length1d):  # loop til the output array is filled up -> one dimensional (600)

                # ACTIVATIONS -> loop through each conv block horizontally
                self.z_values[j][i] = np.sum(input_neurons[:,row:self.filter_size+row, slide:self.filter_size + slide] * self.weights[j]) + self.biases[j]
                self.output[j][i] = sigmoid(self.z_values[j][i])
                slide += self.stride

                if (self.filter_size + slide)-self.stride >= self.width_in:    # wrap indices at the end of each row
                    slide = 0
                    row += self.stride

        self.z_values = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))
        print 'Shape of final conv output: ', self.output.shape
        return self.output, self.z_values


class PoolingLayer(object):

    def __init__(self, input_shape, poolsize = (2,2)):
        '''
        width_in and height_in are the dimensions of the input image
        poolsize is treated as a tuple of filter and stride -> it should work with overlapping pooling
        '''
        self.depth, self.height_in, self.width_in = input_shape
        self.poolsize = poolsize
        self.height_out = (self.height_in - self.poolsize[0])/self.poolsize[1] + 1
        self.width_out = (self.width_in - self.poolsize[0])/self.poolsize[1] + 1      # num of output neurons

        self.output = np.empty((self.depth, self.height_out, self.width_out))
        self.max_indices = np.empty((self.depth, self.height_out, self.width_out, 2))
        print 'Pooling shape (depth,row,col): ', self.output.shape

    def pool(self, input_image):

        self.pool_length1d = self.height_out * self.width_out

        self.output = self.output.reshape((self.depth, self.pool_length1d))
        self.max_indices = self.max_indices.reshape((self.depth, self.pool_length1d, 2))
        
        # for each filter map
        for j in range(self.depth):
            row = 0
            slide = 0
            for i in range(self.pool_length1d):
                toPool = input_image[j][row:self.poolsize[0] + row, slide:self.poolsize[0] + slide]

                self.output[j][i] = np.amax(toPool)                # calculate the max activation
                index = zip(*np.where(np.max(toPool) == toPool))           # save the index of the max
                if len(index) > 1:
                    index = [index[0]]
                index = index[0][0]+ row, index[0][1] + slide
                self.max_indices[j][i] = index

                slide += self.poolsize[1]

                # modify this if stride != filter for poolsize 
                if slide >= self.width_in:
                    slide = 0
                    row += self.poolsize[1]

        self.output = self.output.reshape((self.depth, self.height_out, self.width_out))
        self.max_indices = self.max_indices.reshape((self.depth, self.height_out, self.width_out, 2))
        return self.output

class FullyConnectedLayer(object):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''
    def __init__(self, input_shape, num_output, classify, num_classes = None):
        self.depth, self.height_in, self.width_in = input_shape
        self.num_output = num_output
        self.classify = classify
        self.num_classes = num_classes

        self.weights = np.random.randn(self.num_output, self.depth * self.height_in * self.width_in)
        self.biases = np.random.randn(self.num_output,1)
        # self.weights = np.ones((self.num_output, self.depth * self.height_in * self.width_in))
        # self.biases = np.ones((self.num_output,1))
        
        self.z_values = np.empty((self.num_output))
        self.output = np.empty((self.num_output))

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        # roll out the input image
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        print 'shape of w, input, b: ', self.weights.shape, a.shape, self.biases.shape
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = sigmoid(self.z_values)
        # print self.output
        
        # forwardpass to classification
        if self.classify == True:
            z_vals, final_output = classify(self.output, self.num_output, self.num_classes)
            return self.z_values, self.output, z_vals, final_output
        else:
            return self.z_values, self.output

class Model(object):

    def __init__(self, input_shape, layers):
        '''
        Args:
        Convolutional Layer: shape of input, filter_size, stride, padding, num_filters
        Pooling Layer: shape of input(depth, height_in, width_in), poolsize
        Fully Connected Layer: shape_of_input, num_output, classify = True/False, num_classes (if classify True)
        Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
        '''

        self.input_shape = input_shape

        # e.g. layers: [conv_layer, pool_layer, fc_layer]
        self.layers = layers
        self.setup = []
        first = True
        layerType = ''

        for layer in self.layers:
            # keep track of how many of the same layer you have:
            # if no repetition, don't use numbering -> num = ''
            for key in layer:
                if key[-1].isdigit():
                    num = key[-1]
                    layerType = key
                else:
                    num = ''
                    layerType = key

            if not first:
                new_input_shape = self.setup[-1].output.shape

            if layerType == 'conv_layer{}'.format(num):
                # if it's the first layer, shape = input data's shape
                if first:
                    new_input_shape = self.input_shape
                    first = False
                conv = ConvLayer(
                    new_input_shape,
                    filter_size = layer[layerType]['filter_size'],
                    stride = layer[layerType]['stride'],
                    num_filters = layer[layerType]['num_filters'])
                self.setup.append(conv)

            elif layerType == 'pool_layer{}'.format(num):
                pool = PoolingLayer(
                    new_input_shape,
                    poolsize = layer[layerType]['poolsize'])
                self.setup.append(pool)

            else:
                fc = FullyConnectedLayer(
                    new_input_shape,
                    num_output = layer[layerType]['num_output'],
                    classify = layer[layerType]['classify'],
                    num_classes = layer[layerType]['num_classes'])
                self.setup.append(fc)



    def gradient_descent(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data = None):
        # test for one pic
        plt.imsave('images/training.jpg', training_data[0])
        activations = [([], training_data)]

        # forwardpass
        for layer in self.setup:
            
            if isinstance(layer, ConvLayer) == True:
                conv_input = activations[-1][-1]
                conv_output, conv_z_vals = layer.convolve(conv_input)
                activations.append((conv_z_vals, conv_input, conv_output))

                # this is pretty sweet -> see the image after the convolution
                for i in range(conv_output.shape[0]):
                    plt.imsave('images/cat_conv%s.jpg'%i, conv_output[i])

            elif isinstance(layer, PoolingLayer) == True:
                pool_input = activations[-1][-1]
                pool_output = layer.pool(pool_input)
                activations.append((pool_input, pool_output))

                for i in range(pool_output.shape[0]):
                    plt.imsave('images/pool_pic%s.jpg'%i, pool_output[i])

            else:
                fc_input = activations[-1][-1]
                if not layer.classify:
                    fc_z_vals, fc_output = layer.feedforward(fc_input)
                else:
                    fc_z_vals, fc_output, final_z_vals, final_output = layer.feedforward(fc_input)
                
                activations.append((fc_input, fc_z_vals, fc_output))
                activations.append((fc_output, final_z_vals, final_output))
                print final_output

        # backpropagation
        labels = np.asarray(([1,0])).reshape((2,1))
        delta_w, delta_b, deltas = [],[],[]
        delta_b, delta_w, delta = backprop_final_to_fc(
            prev_activation = activations[-1][0],
            z_vals = activations[-1][1],
            final_output = activations[-1][2],
            y=labels)
        print delta_b, delta_w, delta



# helper functions
###############################################################
def classify(x, num_inputs, num_classes):
    # I. initialize weights and biases!
    w = np.random.randn(num_classes, num_inputs)
    b = np.random.randn(num_classes,1)
    z = np.dot(w,x) + b
    a = sigmoid(z)
    return z, a


# LOSS FUNCTIONS
def cross_entropy(batch_size, output, expected_output):
    return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))

def cross_entropy_prime():
    return 0

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

'''
##################################################################
# test
# delta = np.ones((pool_layer.output.shape[0], pool_layer.output.shape[1], pool_layer.output.shape[2])) * 0.5 
# deltas = None, None, delta
# print delta.shape, '== ? ', pool_layer.max_indices.shape[0:3]
# backprop_pool_to_conv(deltas, conv_output.shape, pool_layer.max_indices)
'''
