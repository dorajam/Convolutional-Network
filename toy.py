# This is the basic idea behind the architecture

import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

from backprop import *


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
        return self.z_values, self.output


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


class Layer(object):

    def __init__(self, input_shape, num_output):
        self.output = np.ones((num_output, 1))
        self.z_values = np.ones((num_output, 1))
        
        
class FullyConnectedLayer(Layer):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''
    def __init__(self, input_shape, num_output):
        super(Layer, self).__init__(input_shape, num_output)
        self.depth, self.height_in, self.width_in = input_shape
        self.num_output = num_output

        self.weights = np.random.randn(self.num_output, self.depth * self.height_in * self.width_in)
        self.biases = np.random.randn(self.num_output,1)

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        # roll out the dimensions
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        # this is shape of (num_outputs, 1)
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = sigmoid(self.z_values)
        
        return self.z_values, self.output

class ClassifyLayer(Layer):
    def __init__(self, num_inputs, num_classes):
        super(Layer, self).__init__(num_inputs, num_classes)
        num_inputs, col = num_inputs
        print 'inputs shape before Final layer: ', num_inputs, col
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_classes, num_inputs)
        self.biases = np.random.randn(self.num_classes,1)

    def classify(self, x):
        self.z_values = np.dot(self.weights,x) + self.biases
        self.output = sigmoid(self.z_values)
        return self.z_values, self.output


class Model(object):

    layer_type_map = {
        'fc_layer': FullyConnectedLayer,
        'final_layer': ClassifyLayer
    }

    def __init__(self, input_shape, layer_config):
        '''
        :param layer_config: list of dicts, outer key is 
        Valid Layer Types:
        Convolutional Layer: shape of input, filter_size, stride, padding, num_filters
        Pooling Layer: shape of input(depth, height_in, width_in), poolsize
        Fully Connected Layer: shape_of_input, num_output, classify = True/False, num_classes (if classify True)
        Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
        '''

        self.input_shape = input_shape
        self._initialize_layers(layer_config)
        self.layer_weight_shapes = [l.weight.shape for l in self.layers]
        self.layer_biases_shapes = [l.biases.shape for l in self.layers]
        print 'Transitions through layers: ',self.layer_transition

    def _initialize_layers(self, layer_config):
        """
        Sets the net's <layer> attribute
        to be a list of Layers (classes from layer_type_map)
        """
        layers = []
        input_shape = self.input_shape
        for layer_spec in layer_config:
            # handle the spec format: {'type': {kwargs}}
            layer_class = self.layer_type_map[layer_spec.keys()[0]]
            layer_kwargs = layer_spec.values()[0]
            layer = layer_class(input_shape, **layer_kwargs)
            input_shape = layer.output.shape
            layers.append(layer)
        self.layers = layers

    def _get_layer_transition(self, inner_ix, outer_ix):
        if inner_ix < 0:
            return 'inputfc'

        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        if (
            isinstance(inner, FullyConnectedLayer) and
            isinstance(outer, ClassifyLayer)
            ):
            return 'fcfinal'

    def feedforward(self, image, label, eta, batch_size, backpropagate = True):
        stride_params, pooling_params = [], []

        # forwardpass
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                # z values are huge, while the fc_output is tiny! large negative vals get penalized to 0!
                fc_input = activations[-1][-1]
                fc_z_vals, fc_output = layer.feedforward(fc_input)
                activations.append((fc_input, fc_z_vals, fc_output))

            elif isinstance(layer, ClassifyLayer):
                final_input = activations[-1][-1]
                final_z_vals, final_output = layer.classify(final_input)
                activations.append((final_input, final_z_vals, final_output))

            else:
                raise NotImplementedError

        if backpropagate:
            return self.backprop(image, label)
        else:
            return activations[-1][-1], None, None

    def backprop(self, image, label):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        print '################# BACKPROP ####################'
        # this is a pointer to the params (weights, biases) on each layer
        num_layers = len(self.layers)
        for l in range(num_layers - 1, -1, -1):
            # the "outer" layer is closer to classification
            # the "inner" layer is closer to input
            inner_layer_ix = l - 1
            outer_layer_ix = l

            layer = self.layers[outer_layer_ix]
            activation = self.layers[inner_layer_ix] if inner_layer_ix > 0 else image

            transition = self._get_layer_transition(
                inner_layer_ix, outer_layer_ix
            )

            print 'Backprop: Inner layer is %d'%(l)

            if transition == 'fcfinal':
                # delta is the one on the final layer
                db, dw, last_delta = backprop_final_to_fc(
                    prev_activation=activation,    # (100,1)
                    z_vals=layer.z_values,         # activations[-1][1],  (10,1)
                    final_output=layer.output,     # activations[-1][2], (10,1)
                    y=label)    # returned delta needs to be UPDATED
                last_weights = layer.weights

            elif transition == 'inputfc':
                # calc delta on the first final layer
                db, dw, _ = backprop_fc_to_input(
                    delta=last_delta,
                    prev_weights=last_weights,    # shape (10,100) this is the weights from the next layer
                    prev_activations=activation,  #(28,28)
                    z_vals=layer.z_values)    # (100,1)

            # print 'delta w, weights shape: ', dw.shape, self.all_weights[weight_count].shape, 'db, biases: ', db.shape, self.all_biases[weight_count].shape
            nabla_b[l], nabla_w[l] = db, dw

        return self.layers[-1].output, nabla_b, nabla_w

      
    def gradient_descent(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data = None):
        training_size = len(training_data)
        if test_data: n_test = len(test_data)
        losses = []

        for epoch in xrange(num_epochs):
            print "Starting epochs"
            start = time.time()
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in xrange(0, training_size, batch_size)]

            for batch in batches:
                loss = self.update_mini_batch(batch, eta)
                losses.append(loss)

                if test_data:
                    print "################## VALIDATE #################"
                    print self.validate(test_data)
                    break
                    print "Epoch {0}: {1} / {2}".format(
                        epoch, self.validate(test_data), n_test)
                    print "Epoch {0} complete".format(epoch)
                    # time
                    timer = time.time() - start
                    print "Estimated time: ", timer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(losses)
        plt.show()

    def update_mini_batch(self, batch, eta):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        batch_size = len(batch)

        for image, label in batch:
            image = image.reshape((1,28,28))
            final_res, delta_b, delta_w = self.feedforward(
                image, label, eta, batch_size, backpropagate=True
            )

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

        ################## print LOSS ############
        error = loss(label, final_res)
        print 'ERROR: ', error
 
        for layer_ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
            layer = self.setup[layer_ix]
            layer.weights -= eta * layer_nabla_w / batch_size
            layer.biases -= eta * layer_nabla_b / batch_size
        return error

        # self.all_weights = [w - eta * dw / batch_size for w,dw in zip(self.all_weights, nabla_w)]
        # self.all_biases= [b - eta * db / batch_size for b,db in zip(self.all_biases, nabla_b)]
        # return error

    def validate(self,data):
        test_results = [(np.argmax(self.feedforward(x,y,None,None, backpropagate=False)),y) for x, y in data]
        return sum(int(x == y) for x, y in test_results) 

# helper functions
###############################################################
def cross_entropy(batch_size, output, expected_output):
    return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))

def cross_entropy_prime():
    return 0

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2
