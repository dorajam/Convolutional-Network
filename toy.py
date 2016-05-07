# This is the basic idea behind the architecture

import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

im = scipy.ndimage.imread('cat.jpg', flatten=True)
print im.shape, type(im)
a = im.shape[0]
b= im.shape[1]
cat = scipy.misc.imresize(im, (a/40,b/40), interp='bilinear', mode=None)
print cat.shape

# normalize
cat = 1 - cat/255
print cat

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
initialize filters in a way that corresponds to the depth of the imagine.
If the input image is of channel 3 (RGB) then your weight vector is n*n*3.
PARAMETERS you'll need: DEPTH (num of filters), STRIDE (slide filter by), ZERO-PADDING(to control the spatial size of the output volumes). Use (Inputs-FilterSize + 2*Padding)/Stride + 1 to calculate your output volume and to decide your hyperparameters'''

DEPTH = 3
STRIDE = 2
# to ensure that the input and output volumes are the same: use P=(F-1)/2 given stride 1.
PADDING = 0
FILTER_SIZE = 5

class ToyNet(object):

    def __init__(self, sizes):
        self.sizes = sizes
        # initialize a list of filters
        self.weights = []
        for i in range(DEPTH):
            self.weights.append([np.random.randn(FILTER_SIZE, FILTER_SIZE)])
        self.biases = np.random.rand(DEPTH,1)
        self.activations = []

    def convolve(self, input_neurons):
        output_dim1 = (input_neurons.shape[0] - FILTER_SIZE + 2*PADDING)/STRIDE + 1
        output_dim2 =  (input_neurons.shape[1] - FILTER_SIZE + 2*PADDING)/STRIDE + 1

        for i in range(DEPTH):
            self.activations.append(np.empty((output_dim1 * output_dim2)))

        print 'shape of input: ', input_neurons.shape
        print 'shape of output: ','(', output_dim1,',', output_dim2, ')'

        for i in range(DEPTH):
            slide = 0
            k = 0
            row = 0
            print self.activations[i].shape[0]    # one dimensional
            while k < self.activations[i].shape[0]:  # til the output array is filled up -> one dimensional (600)
                if FILTER_SIZE + slide < input_neurons.shape[0]:
                    self.activations[i][k] = sigmoid(np.sum(input_neurons[slide:FILTER_SIZE + slide,row:FILTER_SIZE+row] * self.weights[i][0]) + self.biases[i])
                    slide += STRIDE
                else:
                    self.activations[i][k] = sigmoid(np.sum(input_neurons[slide:FILTER_SIZE + slide,row:FILTER_SIZE+row] * self.weights[i][0]) + self.biases[i])
                    slide = 0
                    row += STRIDE
                # if slide
                k += 1

            self.activations[i] = self.activations[i].reshape((output_dim1, output_dim2))
        # print self.activations[0]



class PoolingLayer(object):

    def __init__(self, width_in, height_in, depth, poolsize = (2,2)):
        '''
        width_in and height_in are the dimensions of the input image
        poolsize is treated as a tuple of filter and stride -> it should work with overlapping pooling
        '''
        self.width_in = width_in
        self.height_in = height_in
        self.depth = depth
        self.poolsize = poolsize
        self.width_out = (self.width_in - self.poolsize[0])/self.poolsize[1] + 1
        self.height_out = (self.height_in - self.poolsize[0])/self.poolsize[1] + 1

        # initialize empty output matrix
        self.output = np.empty((self.depth * self.width_out * self.height_out))
        self.max_indeces = np.empty((self.depth * self.width_out * self.height_out, 2))
        print self.output.shape

    def pool(self, input_image):
        k = 0

        # for each filter map
        for j in range(self.depth):
            row = 0
            slide = 0
            for i in range(self.width_out * self.height_out):
                toPool = input_image[j][slide:self.poolsize[0] + slide,row:self.poolsize[0] + row]

                self.output[k] = np.amax(toPool)                # calculate the max activation
                index = zip(*np.where(np.max(toPool) == toPool))           # save the index of the max
                if len(index) > 1:
                    index = [index[0]]
                index = index[0][0]+ row, index[0][1] + slide
                self.max_indeces[k] = index

                slide += self.poolsize[1]

                if slide >= self.width_in:
                    slide = 0
                    row += self.poolsize[1]
                k += 1

        self.output = self.output.reshape((self.depth, self.width_out, self.height_out))
        self.max_indeces = self.max_indeces.reshape((self.depth, self.width_out, self.height_out, 2))
        # print 'AFTER RESHPAING:', self.output
        # print self.max_indeces

class FullyConnectedLayer(object):

    def __init__(self, width_in, height_in, depth, width_out, height_out):
        self.width_in = width_in
        self.height_in = height_in
        self.depth = depth
        self.width_out = width_out
        self.height_out = height_out

        self.weights = np.random.randn(self.width_in, self.height_in, self.depth, self.width_out, self.height_out)
        self.biases = np.random.randn(self.width_out, self.height_out)

    def feedforward(self, input_matrix):
        print self.weights.shape
        for b,w in zip(self.biases, self.weights):
            print b.shape,w.shape
            break 



def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))



net = ToyNet([cat.shape[0]*cat.shape[1]])
print 'yooooo', net.sizes[0]
net.convolve(cat)
pool_layer = PoolingLayer(net.activations[0].shape[0], net.activations[0].shape[1], len(net.activations)) # only implemented for the first depth layer
pool_layer.pool(net.activations)
fc_layer = FullyConnectedLayer(pool_layer.output.shape[1], pool_layer.output.shape[2], pool_layer.output[0], 10, 1)
fc_layer.feedforward(pool_layer.output)
