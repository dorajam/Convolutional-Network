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
cat = 1.0 - cat/255.0
# print cat[30:40]

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
STRIDE = 1
# to ensure that the input and output volumes are the same: use P=(F-1)/2 given stride 1.
PADDING = 0
FILTER_SIZE = 2

class ToyNet(object):

    def __init__(self, sizes):
        self.sizes = sizes
        # initialize a list of filters
        self.weights = []
        for i in range(DEPTH):
            self.weights.append([np.random.randn(FILTER_SIZE, FILTER_SIZE)])
        self.biases = np.random.rand(DEPTH,1)
        self.activations = None

    def convolve(self, input_neurons):
        '''
        Assume input image to be of channel one!
        '''
        output_dim1 = (input_neurons.shape[0] - FILTER_SIZE + 2*PADDING)/STRIDE + 1        # num of rows
        output_dim2 =  (input_neurons.shape[1] - FILTER_SIZE + 2*PADDING)/STRIDE + 1       # num of cols

        self.activations = np.zeros((DEPTH, output_dim1 * output_dim2))

        # for i in range(DEPTH):
        #     self.activations.append(np.empty((output_dim1 * output_dim2)))

        print 'shape of input (rows,cols): ', input_neurons.shape
        print 'shape of output (rows, cols): ','(', output_dim1,',', output_dim2, ')'
        act_length =  self.activations[0].shape[0]
        print 'shape of unrolled convolution output: ', act_length   # one dimensional

        for j in range(DEPTH):
            slide = 0
            row = 0

            for i in range(act_length):  # loop til the output array is filled up -> one dimensional (600)

                # ACTIVATIONS -> loop through each 2x2 block horizontally
                self.activations[j][i] = sigmoid(np.sum(input_neurons[row:FILTER_SIZE+row, slide:FILTER_SIZE + slide] * self.weights[j][0]) + self.biases[j])
                slide += STRIDE

                if (FILTER_SIZE + slide)-STRIDE >= input_neurons.shape[1]:    # wrap indeces at the end of each row
                    slide = 0
                    row += STRIDE
        self.activations = self.activations.reshape((DEPTH, output_dim1, output_dim2))
        print 'Shape of final conv output: ', self.activations.shape
        return self.activations


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
        self.width_out = (self.width_in - self.poolsize[0])/self.poolsize[1] + 1      # num of output neurons
        self.height_out = (self.height_in - self.poolsize[0])/self.poolsize[1] + 1
        print 'Pooling shape (row,col): ', self.height_out, self.width_out

        # initialize empty output matrix
        self.output = np.empty((self.depth, self.width_out * self.height_out))
        self.max_indeces = np.empty((self.depth, self.width_out * self.height_out, 2))

    def pool(self, input_image):
        k = 0
        
        # for each filter map
        for j in range(self.depth):
            row = 0
            slide = 0
            for i in range(self.width_out * self.height_out):
                toPool = input_image[j][row:self.poolsize[0] + row, slide:self.poolsize[0] + slide]

                self.output[j][k] = np.amax(toPool)                # calculate the max activation
                index = zip(*np.where(np.max(toPool) == toPool))           # save the index of the max
                if len(index) > 1:
                    index = [index[0]]
                index = index[0][0]+ row, index[0][1] + slide
                self.max_indeces[j][k] = index

                slide += self.poolsize[1]

                # modify this if stride != filter for poolsize 
                if slide >= self.width_in:
                    slide = 0
                    row += self.poolsize[1]
                k += 1
#                 print 'matrix: ', toPool,'max', self.output[j][k-1]
#                 print 'index: ', self.max_indeces[j][k-1]
#                 if k > 10:
#                     break

        self.output = self.output.reshape((self.depth, self.height_out, self.width_out))
        self.max_indeces = self.max_indeces.reshape((self.depth, self.height_out, self.width_out, 2))
#         print 'AFTER RESHPAING:', self.output


class FullyConnectedLayer(object):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''
    def __init__(self, depth, height_in, width_in, num_output, num_classes):
        self.width_in = width_in
        self.height_in = height_in
        self.depth = depth
        self.num_output = num_output
        self.num_classes = num_classes

        self.weights = np.random.randn(self.num_output, self.depth * self.height_in * self.width_in)
        self.biases = np.random.randn(self.num_output,1)
        # self.weights = np.ones((self.num_output, self.depth * self.height_in * self.width_in))
        # self.biases = np.ones((self.num_output,1))
        
        self.output = np.empty((self.num_output))
        self.final_output = np.empty((self.num_classes))

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        print 'shape of w, input, b: ', self.weights.shape, a.shape, self.biases.shape
        self.output = sigmoid(np.dot(self.weights, a) + self.biases)
        print self.output
        
        # forwardpass to classification
        self.final_output = classify(self.output, self.num_output, self.num_classes)
        return self.final_output


# helper functions
###############################################################
def classify(x, num_inputs, num_classes):
    # I. initialize weights and biases!
    w = np.random.randn(num_classes, num_inputs)
    b = np.random.randn(num_classes,1)
    return sigmoid(np.dot(w,x) + b)

def cross_entropy(batch_size, output, expected_output):
    return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))

def cross_entropy_prime():
    return 0

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

# setting up
#################################################################
net = ToyNet([cat.shape[0]*cat.shape[1]])
print 'yooooo', net.sizes[0]
conv_output = net.convolve(cat)
print type(conv_output)
# this is pretty sweet -> see the image after the convolution
for i in range(conv_output.shape[0]):
    plt.imsave('cat_conv%s.jpg'%i, conv_output[i])

# TODO: implement for all activations!
pool_layer = PoolingLayer(12, 12, 1) # only implemented for the first depth layer
# # pool_layer.pool(test)

# test = 1 - test /144
# test = test.reshape((3*12*4,1))
# fc = FullyConnectedLayer(3,12,4,10,2)
# fc.feedforward(test)



# testing
##################################################################
# a = np.arange(8).reshape((2*2*2,1))
# w = np.ones(16).reshape((2,2,2,2))
# o = np.arange(2).reshape((2,1))

# fc = FullyConnectedLayer(2,2,2,2,0)
# fc.feedforward(a)
