# This is the basic idea behind the architecture

import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

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

        self.weights = np.random.randn(self.num_filters, self.filter_size, self.filter_size)
        self.biases = np.random.rand(self.num_filters,1)

        self.output_dim1 = (self.height_in - self.filter_size + 2*self.padding)/self.stride + 1        # num of rows
        self.output_dim2 =  (self.width_in - self.filter_size + 2*self.padding)/self.stride + 1         # num of cols
        
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))

        print 'shape of input (depth, rows,cols): ', input_shape
        print 'shape of convolutional layer (depth, rows, cols): ', self.output.shape


    def convolve(self, input_neurons):
        '''
        Pass in the actual input data and do the convolution.
        Returns: sigmoid activation matrix after convolution 
        '''

        # roll out activations
        self.output = np.zeros((self.num_filters, self.output_dim1 * self.output_dim2))
        
        act_length1d =  self.output.shape[1]
        print 'shape of each rolled feature map: ', act_length1d   # one dimensional

        for j in range(self.num_filters):
            slide = 0
            row = 0

            for i in range(act_length1d):  # loop til the output array is filled up -> one dimensional (600)

                # ACTIVATIONS -> loop through each conv block horizontally
                self.output[j][i] = sigmoid(np.sum(input_neurons[row:self.filter_size+row, slide:self.filter_size + slide] * self.weights[j][0]) + self.biases[j])
                slide += self.stride

                if (self.filter_size + slide)-self.stride >= self.width_in:    # wrap indices at the end of each row
                    slide = 0
                    row += self.stride

        self.output = self.output.reshape((self.filter_size, self.output_dim1, self.output_dim2))
        print 'Shape of final conv output: ', self.output.shape
        return self.output


class PoolingLayer(object):

    def __init__(self, shape_of_input, poolsize = (2,2)):
        '''
        width_in and height_in are the dimensions of the input image
        poolsize is treated as a tuple of filter and stride -> it should work with overlapping pooling
        '''
        self.depth, self.height_in, self.width_in = shape_of_input
        self.poolsize = poolsize
        self.height_out = (self.height_in - self.poolsize[0])/self.poolsize[1] + 1
        self.width_out = (self.width_in - self.poolsize[0])/self.poolsize[1] + 1      # num of output neurons
        print 'Pooling shape (depth,row,col): ', self.depth, self.height_out, self.width_out

    def pool(self, input_image):

        self.pool_length1d = self.height_out * self.width_out

        self.output = np.empty((self.depth, self.pool_length1d))
        self.max_indices = np.empty((self.depth, self.pool_length1d, 2))
        
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
#                 print 'matrix: ', toPool,'max', self.output[j][k-1]
#                 print 'index: ', self.max_indices[j][k-1]
#                 if k > 10:
#                     break

        self.output = self.output.reshape((self.depth, self.height_out, self.width_out))
        self.max_indices = self.max_indices.reshape((self.depth, self.height_out, self.width_out, 2))
        # print self.max_indices
#         print 'AFTER RESHPAING:', self.output


class FullyConnectedLayer(object):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''
    def __init__(self, depth, height_in, width_in, num_output, classify, num_classes = None):
        self.width_in = width_in
        self.height_in = height_in
        self.depth = depth
        self.num_output = num_output
        self.classify = classify
        self.num_classes = num_classes

        self.weights = np.random.randn(self.num_output, self.depth * self.height_in * self.width_in)
        self.biases = np.random.randn(self.num_output,1)
        # self.weights = np.ones((self.num_output, self.depth * self.height_in * self.width_in))
        # self.biases = np.ones((self.num_output,1))
        
        self.output = np.empty((self.num_output))
        self.final_output = None

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        print 'shape of w, input, b: ', self.weights.shape, a.shape, self.biases.shape
        self.output = sigmoid(np.dot(self.weights, a) + self.biases)
        # print self.output
        
        # forwardpass to classification
        if self.classify == True:
            self.final_output = classify(self.output, self.num_output, self.num_classes)
            return self.final_output
        else:
            return self.output

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

        for layer in self.layers:
            # import ipdb; ipdb.set_trace()
            # keep track of how many of the same layer you have:
            # if no repetition, don't use numbering -> num = ''
            for key in layer:
                if key[-1].isdigit():
                    num = key[-1]
                    layer = key
                else:
                    num = ''
                    layer = key

            # check what type the layer is and take args based on the unique keys
            layer_type = ['conv_layer{}'.format(num), 'pool_layer{}'.format(num), 'fc_layer{}'.format(num)]
            if self.setup != []:
                new_input_shape = self.setup[-1].output.shape

            if layer == layer_type[0]:
                # if it's the first layer, shape = input data's shape
                if first:
                    conv = ConvLayer(
                        input_shape = self.input_shape,
                        filter_size = layers[layer_type[0]]['filter_size'],
                        stride = layers[layer_type[0]]['stride'],
                        num_filters = layers[layer_type[0]]['num_filters'])
                    first = False
                else:
                    conv = ConvLayer(
                        new_input_shape,
                        filter_size = layers[layer_type[0]]['filter_size'],
                        stride = layers[layer_type[0]]['stride'],
                        num_filters = layers[layer_type[0]]['num_filters'])
                self.setup.append(conv)
            elif layer == layer_type[1]:
                pool = PoolingLayer(
                    new_input_shape,
                    poolsize = layers[layer_type[1]]['poolsize'])
                self.setup.append(pool)
            else:
                fc = FullyConnectedLayer(
                    new_input_shape,
                    num_output = layers[layer_type[2]]['num_output'],
                    classify = layers[layer_type[2]]['classify'],
                    num_classes = layers[layer_type[2]]['num_classes'])
                self.setup.append(fc)

            


        def gradient_descent(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data = None ):
            # test for one pic
            print 'hello'







# helper functions
###############################################################
def classify(x, num_inputs, num_classes):
    # I. initialize weights and biases!
    w = np.random.randn(num_classes, num_inputs)
    b = np.random.randn(num_classes,1)
    return sigmoid(np.dot(w,x) + b)


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
# setting up
#################################################################
net = ConvLayer([cat.shape[0]*cat.shape[1]])    # make sure this works for RGB, too
print 'yooooo', net.sizes[0]
conv_output = net.convolve(cat)

# this is pretty sweet -> see the image after the convolution
for i in range(conv_output.shape[0]):
    plt.imsave('cat_conv%s.jpg'%i, conv_output[i])

# TODO: implement for all activations!
pool_layer = PoolingLayer(conv_output.shape[0], conv_output.shape[1], conv_output.shape[2])
pool_layer.pool(conv_output)
for i in range(pool_layer.output.shape[0]):
    plt.imsave('pool_pic%s.jpg'%i, pool_layer.output[i])



##################################################################
# test
# delta = np.ones((pool_layer.output.shape[0], pool_layer.output.shape[1], pool_layer.output.shape[2])) * 0.5 
# deltas = None, None, delta
# print delta.shape, '== ? ', pool_layer.max_indices.shape[0:3]
# backprop_pool_to_conv(deltas, conv_output.shape, pool_layer.max_indices)


# testing
##################################################################
# a = np.arange(8).reshape((2*2*2,1))
# w = np.ones(16).reshape((2,2,2,2))
# o = np.arange(2).reshape((2,1))

# fc = FullyConnectedLayer(2,2,2,2,0)
# fc.feedforward(a)
'''
