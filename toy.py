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


class Layer(object):

    def __init__(self, input_shape, num_output):
        self.output = np.ones((num_output, 1))
        self.z_values = np.ones((num_output, 1))
        
        
class FullyConnectedLayer(Layer):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''
    def __init__(self, input_shape, num_output):
        super(FullyConnectedLayer, self).__init__(input_shape, num_output)
        self.depth, self.height_in, self.width_in = input_shape
        self.num_output = num_output

        self.weights = np.random.randn(self.num_output, self.depth, self.height_in, self.width_in)
        self.biases = np.random.randn(self.num_output,1)

    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        # roll out the dimensions
        self.weights = self.weights.reshape((self.num_output, self.depth * self.height_in * self.width_in))
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        # this is shape of (num_outputs, 1)
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = sigmoid(self.z_values)
        self.weights = self.weights.reshape((self.num_output, self.depth, self.height_in, self.width_in))
        
class ClassifyLayer(Layer):
    def __init__(self, num_inputs, num_classes):
        super(ClassifyLayer, self).__init__(num_inputs, num_classes)
        num_inputs, col = num_inputs
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_classes, num_inputs)
        self.biases = np.random.randn(self.num_classes,1)

    def classify(self, x):
        self.z_values = np.dot(self.weights,x) + self.biases
        self.output = sigmoid(self.z_values)


class Model(object):

    layer_type_map = {
        'fc_layer': FullyConnectedLayer,
        'final_layer': ClassifyLayer,
        'conv_layer': ConvLayer,
        'pool_layer': PoolingLayer
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
        self.layer_weight_shapes = [l.weights.shape for l in self.layers if not isinstance(l,PoolingLayer)]
        self.layer_biases_shapes = [l.biases.shape for l in self.layers if not isinstance(l,PoolingLayer)]

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
        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        # either input to FC or pool to FC -> going from 3d matrix to 1d
        if (
            (inner_ix < 0 or isinstance(inner, PoolingLayer)) and 
            isinstance(outer, FullyConnectedLayer)
            ):
            return '3d_to_1d'
        # going from 3d to 3d matrix -> either input to conv or conv to conv
        if (
            (inner_ix < 0 or isinstance(inner, ConvLayer)) and 
            isinstance(outer, ConvLayer)
            ):
            return 'to_conv'
        if (
            isinstance(inner, FullyConnectedLayer) and
            (isinstance(outer, ClassifyLayer) or isinstance(outer, FullyConnectedLayer))
            ):
            return '1d_to_1d'
        if (
            isinstance(inner, ConvLayer) and
            isinstance(outer, PoolingLayer)
            ):
            return 'conv_to_pool'

        raise NotImplementedError

    def feedforward(self, image):
        prev_activation = image

        # forwardpass
        for layer in self.layers:
            input_to_feed = prev_activation

            if isinstance(layer, FullyConnectedLayer):
                # z values are huge, while the fc_output is tiny! large negative vals get penalized to 0!
                layer.feedforward(input_to_feed)

            elif isinstance(layer, ConvLayer):
                layer.convolve(input_to_feed)
                for i in range(layer.output.shape[0]):
                    plt.imsave('images/cat_conv%d.jpg'%i, layer.output[i])
                for i in range(layer.weights.shape[0]):
                    plt.imsave('images/filter_conv%s.jpg'%i, layer.weights[i].reshape((5,5)))

            elif isinstance(layer, PoolingLayer):
                layer.pool(input_to_feed)
                for i in range(layer.output.shape[0]):
                    plt.imsave('images/pool_pic%s.jpg'%i, layer.output[i])

            elif isinstance(layer, ClassifyLayer):
                layer.classify(input_to_feed)

            else:
                raise NotImplementedError

            prev_activation = layer.output

        final_activation = prev_activation
        return final_activation

    def backprop(self, image, label):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        # set first params on the final layer
        final_output = self.layers[-1].output
        last_delta = (final_output - label) * sigmoid_prime(self.layers[-1].z_values)
        last_weights = None
        final=True

        num_layers = len(self.layers)
        # import ipdb;ipdb.set_trace()

        for l in range(num_layers - 1, -1, -1):
            # the "outer" layer is closer to classification
            # the "inner" layer is closer to input
            inner_layer_ix = l - 1
            if (l-1) <0:
                inner_layer_ix = 0
            outer_layer_ix = l

            layer = self.layers[outer_layer_ix]
            activation = self.layers[inner_layer_ix].output if inner_layer_ix >= 0 else image

            transition = self._get_layer_transition(
                inner_layer_ix, outer_layer_ix
            )

            # inputfc = poolfc
            # fc to fc = fc to final
            # conv to conv -> input to conv
            # conv to pool -> unique

            if transition == '1d_to_1d':   # final to fc, fc to fc
                db, dw, last_delta = backprop_1d_to_1d(
                    delta = last_delta,
                    prev_weights=last_weights,
                    prev_activations=activation,
                    z_vals=layer.z_values,
                    final=final)
                final = False

            elif transition == '3d_to_1d':
                if l==0:
                    activation = image
                # calc delta on the first final layer
                db, dw, last_delta = backprop_1d_to_3d(
                    delta=last_delta,
                    prev_weights=last_weights,    # shape (10,100) this is the weights from the next layer
                    prev_activations=activation,  #(28,28)
                    z_vals=layer.z_values)    # (100,1)
                # layer.weights = layer.weights.reshape((layer.num_output, layer.depth, layer.height_in, layer.width_in))

            # pool to conv layer
            elif transition == 'conv_to_pool':
                # no update for dw,db => only backprops the error            
                last_delta = backprop_pool_to_conv(
                    delta = last_delta,
                    prev_weights = last_weights,
                    input_from_conv = activation,
                    max_indices = layer.max_indices,
                    poolsize = layer.poolsize,
                    pool_output = layer.output)

            # conv to conv layer
            elif transition == 'to_conv':
                # weights passed in are the ones between conv to conv
                # update the weights and biases
                activation = image
                last_weights = layer.weights
                db,dw = backprop_to_conv(
                    delta = last_delta,
                    weight_filters = last_weights,
                    stride = layer.stride,
                    input_to_conv = activation,
                    prev_z_vals = layer.z_values)
            else:
                pass

            if transition != 'conv_to_pool':
                # print 'nablasb, db,nabldw, dw, DELTA', nabla_b[inner_layer_ix].shape, db.shape, nabla_w[inner_layer_ix].shape, dw.shape, last_delta.shape
                nabla_b[inner_layer_ix], nabla_w[inner_layer_ix] = db, dw
                last_weights = layer.weights

        return self.layers[-1].output, nabla_b, nabla_w

      
    def gradient_descent(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data = None):
        training_size = len(training_data)
        if test_data: n_test = len(test_data)

        mean_error = []
        correct_res = []

        for epoch in xrange(num_epochs):
            print "Starting epochs"
            start = time.time()
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in xrange(0, training_size, batch_size)]
            losses = 0

            for batch in batches:
                loss = self.update_mini_batch(batch, eta)
                losses+=loss
            mean_error.append(round(losses/batch_size,2))
            print mean_error

            if test_data:
                print "################## VALIDATE #################"
                res = self.validate(test_data)
                correct_res.append(res)
                print "Epoch {0}: {1} / {2}".format(
                    epoch, self.validate(test_data), n_test)
                print "Epoch {0} complete".format(epoch)
                # time
                timer = time.time() - start
                print "Estimated time: ", timer
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(correct_res)
        plt.show()

    def update_mini_batch(self, batch, eta):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        batch_size = len(batch)

        for image, label in batch:
            image = image.reshape((1,28,28))
            _ = self.feedforward(image)
            final_res, delta_b, delta_w = self.backprop(image, label)

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

        ################## print LOSS ############
        error = loss(label, final_res)
        
        num =0
        weight_index = []
        for layer in self.layers:
            if not isinstance(layer,PoolingLayer):
                weight_index.append(num)
            num+=1

        for ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
            layer = self.layers[weight_index[ix]]
            layer.weights -= eta * layer_nabla_w / batch_size
            layer.biases -= eta * layer_nabla_b / batch_size
        return error

    def validate(self,data):
        data = [(im.reshape((1,28,28)),y) for im,y in data]
        test_results = [(np.argmax(self.feedforward(x)),y) for x, y in data]
        return sum(int(x == y) for x, y in test_results) 

# helper functions
###############################################################
def cross_entropy(batch_size, output, expected_output):
    return (-1/batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1-output))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2
