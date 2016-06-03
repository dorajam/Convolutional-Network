# This is the basic idea behind the architecture

import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

from backprop import *

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

        # print 'shape of input (depth, rows,cols): ', input_shape
        # print 'shape of convolutional layer (depth, rows, cols): ', self.output.shape


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
        # print 'Shape of final conv output: ', self.output.shape
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
        # print 'Pooling shape (depth,row,col): ', self.output.shape

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
        if (self.classify):
            self.final_weights = np.zeros((self.num_output, self.num_classes))
            self.final_biases = (self.num_classes,1)
        
        self.z_values = np.ones((1, self.num_output, 1))
        self.output = np.ones((1, self.num_output, 1))

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        # roll out the dimensions
        # self.weights = self.weights.reshape((self.num_output, self.depth * self.height_in * self.width_in))
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        # print 'shape of w, input, b: ', self.weights.shape, a.shape, self.biases.shape
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = sigmoid(self.z_values)
        # self.weights = self.weights.reshape((self.num_output, self.depth, self.height_in, self.width_in))
        
        # forwardpass to classification
        if self.classify == True:
            w,b, z_vals, final_output = classify(self.output, self.num_output, self.num_classes)
            self.final_weights = w
            self.final_biases = b
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
        self.layer_transition = ['']
        
        # e.g. layers: [conv_layer, pool_layer, fc_layer]
        self.layers = layers
        self.setup = []
        self.all_weights = []
        self.all_biases = []
        first = True
        self.layerType = ''

        for layer in self.layers:
            # keep track of how many of the same layer you have:
            # if no repetition, don't use numbering -> num = ''
            for key in layer:
                if key[-1].isdigit():
                    num = key[-1]
                    self.layerType = key
                else:
                    num = ''
                    self.layerType = key

            if not first:
                new_input_shape = self.setup[-1].output.shape

            if self.layerType == 'conv_layer{}'.format(num):
                name = 'conv'
                # if it's the first layer, shape = input data's shape
                if first:
                    new_input_shape = self.input_shape
                    first = False
                conv = ConvLayer(
                    new_input_shape,
                    filter_size = layer[self.layerType]['filter_size'],
                    stride = layer[self.layerType]['stride'],
                    num_filters = layer[self.layerType]['num_filters'])
                self.setup.append(conv)
                self.all_weights.append(conv.weights)
                self.all_biases.append(conv.biases)

            elif self.layerType == 'pool_layer{}'.format(num):
                name = 'pool'
                pool = PoolingLayer(
                    new_input_shape,
                    poolsize = layer[self.layerType]['poolsize'])
                self.setup.append(pool)

            else:
                name = 'fc'
                fc = FullyConnectedLayer(
                    new_input_shape,
                    num_output = layer[self.layerType]['num_output'],
                    classify = layer[self.layerType]['classify'],
                    num_classes = layer[self.layerType]['num_classes'])
                self.setup.append(fc)
                self.all_weights.append(fc.weights)
                self.all_biases.append(fc.biases)
                if fc.classify == True:
                    self.all_weights.append(fc.final_weights)
                    self.all_biases.append(fc.final_biases)

            # store the names of the layers
            self.layer_transition[-1] = self.layer_transition[-1] + name
            self.layer_transition.append(name)
        print self.layer_transition


    def feedforward(self, data, eta, batch_size, backpropagate = True):
        print len(data)
        label = data[1]
        image = data[0]
        # image = data[0].reshape((1,28,28))
        activations = [([], image)]
        all_delta_w, all_delta_b = [], []
        stride_params, pooling_params = [], []

        # forwardpass
        for layer in self.setup:
            if isinstance(layer, ConvLayer) == True:
                conv_input = activations[-1][-1]
                conv_output, conv_z_vals = layer.convolve(conv_input)
                activations.append((conv_input, conv_z_vals, conv_output))
                stride_params.append(layer.stride)
                
                # this is pretty sweet -> see the image after the convolution
                for i in range(conv_output.shape[0]):
                    plt.imsave('images/cat_conv%s.jpg'%i, conv_output[i])

            elif isinstance(layer, PoolingLayer) == True:
                pool_input = activations[-1][-1]
                pool_output = layer.pool(pool_input)
                activations.append((pool_input, layer.max_indices, pool_output))
                pooling_params.append(layer.poolsize)
                
                for i in range(pool_output.shape[0]):
                    plt.imsave('images/pool_pic%s.jpg'%i, pool_output[i])
                    
            else:
                # z values are huge, while the fc_output is tiny! large negative vals get penalized to 0!
                fc_input = activations[-1][-1]
                if not layer.classify:
                    fc_z_vals, fc_output = layer.feedforward(fc_input)
                    activations.append((fc_input, fc_z_vals, fc_output))
                else:
                    fc_z_vals, fc_output, final_z_vals, final_output = layer.feedforward(fc_input)
                    activations.append((fc_input, fc_z_vals, fc_output))
                    activations.append((fc_output, final_z_vals, final_output))

        def backprop():
            # import ipdb;ipdb.set_trace()
            for i in range(self.all_weights[0].shape[0]):
                im =  self.all_weights[0][i].reshape((3,3))
                plt.imsave('images/fitlers%s.jpg'%i,im)

            print '################# BACKPROP ####################'

            # this is a pointer to the params (weights, biases) on each layer
            weight_count = len(self.all_weights) - 1

            for l in range(len(self.layer_transition)-1,-1, -1):
                transition = self.layer_transition[l]
                # print 'This is the %dth layer'%(l+1)

                # final layer
                if transition == 'fc':
                    # delta is the one on the final layer
                    db, dw, delta = backprop_final_to_fc(
                        prev_activation = activations[l+1][0],
                        z_vals = activations[l+1][1],
                        final_output = activations[l+1][2],
                        y=label)    # returned delta needs to be UPDATED
                                    
                # fc to fc layer
                elif transition == 'fcfc':
                    # calc delta on the first final layer
                    db,dw, delta = calc_gradients(
                        delta = delta,
                        prev_weights = self.all_weights[weight_count],
                        prev_activations = activations[l+1][0],
                        z_vals = activations[l+1][1])
                    weight_count -= 1     # set pointer to the weights on prev layer
                                    
                # fc to pool layer
                elif transition == 'poolfc':
                    # calc delta on the fc layer
                    db,dw, delta = backprop_fc_to_pool(
                        delta = delta,
                        prev_weights = self.all_weights[weight_count],
                        prev_activations = activations[l+1][0],
                        z_vals = activations[l+1][1])
                    weight_count -= 1     # set pointer to the weights on prev layer

                # pool to conv layer
                elif transition == 'convpool':
                    # delta is the one on the conv layer
                    # no update here!
                    delta = backprop_pool_to_conv(
                        delta = delta,
                        prev_weights = self.all_weights[weight_count],
                        input_from_conv = activations[l+1][0],
                        max_indices = activations[l+1][1],
                        poolsize = pooling_params[-1],
                        pool_output = activations[l+1][2])
                    pooling_params.pop()
                    weight_count -= 1     # set pointer to the weights on prev layer

                # conv to conv layer
                elif transition == 'convconv':
                    # weights passed in are the ones between conv to conv
                    # update the weights and biases 
                    db,dw = backprop_conv_to_conv(
                        delta = delta,
                        weight_filters = self.all_weights[weight_count],
                        stride = stride_params[-1],
                        input_to_conv = activations[l+1][0],
                        prev_z_vals = activations[l+1][1])
                    stride_params.pop()
                    update(weight_count, eta, self.all_weights[weight_count], self.all_biases[weight_count], dw,db, batch_size = 1)
                    weight_count -= 1     # set pointer to the weights on prev layer

                # beginning 
                else:
                    db,dw = backprop_conv_to_conv(
                        delta = delta,
                        weight_filters = self.all_weights[weight_count],
                        stride = stride_params[-1],
                        input_to_conv = activations[l+1][0],
                        prev_z_vals = activations[l+1][1])
                    stride_params.pop()
                    self.update(weight_count, eta, self.all_weights[weight_count], self.all_biases[weight_count], dw,db, batch_size = 1)

                if (transition != 'convpool') and (transition !='convconv') and (transition !='conv'):
                    # print 'delta,dw, weights shape: ',delta.shape, dw.shape, self.all_weights[weight_count].shape
                    self.update(weight_count, eta, self.all_weights[weight_count], self.all_biases[weight_count], dw,db, batch_size = 1)

        if backpropagate == True:
            backprop()
        else:
            return activations[-1][-1]
      
    def gradient_descent(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data = None):
        training_size = len(training_data)
        if test_data: n_test = len(test_data)

        for epoch in xrange(num_epochs):
            print "Starting epochs"
            start = time.time()
            # random.shuffle(training_data)
            # batches = [training_data[k:k + batch_size] for k in xrange(0, training_size, batch_size)]
            # batches = training_data

            # for batch in batches:
            # for image_tuple in training_data:
            for i in range(1):
                image_tuple = training_data
                print image_tuple[0].shape
                # test for one pic
                # plt.imsave('images/training.jpg', image_tuple[0])
                
                self.feedforward(image_tuple, eta, batch_size, backpropagate=True)
            if test_data:
                print "################## VALIDATE #################"
                print "Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluate(test_data), n_test)
                print "Epoch {0} complete".format(epoch)
                # time
                timer = time.time() - start
                print "Estimated time: ", timer

    def update(self,num, eta, weights, biases, dw, db, batch_size=1):
        # print weights.shape, dw.shape, biases.shape,db.shape
        print num
        if num == len(self.all_weights)-1:
            weights = weights.transpose()
            self.all_weights[num] = (weights - eta * dw/batch_size).transpose()
        else:
            self.all_weights[num] = weights - eta * dw/batch_size
            self.all_biases[num] = biases - eta * db


    def evaluate(self,data):
        test_results = [(np.argmax(self.feedforward((x,y),None,None, backpropagate=False)),y) for x, y in data]
        return sum(int(x == y) for x, y in test_results) 

# helper functions
###############################################################
def classify(x, num_inputs, num_classes):
    # I. initialize weights and biases!
    w = np.random.randn(num_classes, num_inputs)
    b = np.random.randn(num_classes,1)
    z = np.dot(w,x) + b
    a = sigmoid(z)
    return w,b,z, a


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
