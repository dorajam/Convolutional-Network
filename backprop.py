# Dora Jambor
# May 2016
# helper functions for backpropagation

import numpy as np


# backpropagation
##############################################################
# I. Go from last layer to FC layer
def backprop_final_to_fc(prev_activation, z_vals, final_output, y):
    delta = (final_output - y) * sigmoid_prime(z_vals)
    delta_b = delta
    delta_w = np.dot(delta, prev_activation.transpose())
    return delta_b, delta_w, delta

def backprop_fc_to_pool(deltas, weights, fc_input, prev_z_vals):
    _, _, delta = deltas
    return calc_gradients(delta, weights, fc_input, prev_z_vals)

def backprop_pool_to_conv(deltas, conv_shape, max_indices):
    depth, height, width = conv_shape
    delta_w, delta_b, delta = deltas
    delta_new = np.zeros((depth, height, width))
    depth = max_indices.shape[0]
    
    # shape of delta should be the same as max_indeces
    if depth != delta.shape[0]:
        raise Exception('Pooling shape is not aligned with deltas')

    # roll out the delta matrix from pooling layer
    pool_height, pool_width = delta.shape[1], delta.shape[2]
    delta = delta.reshape((depth, pool_height * pool_width))
    # same for the max index matrix
    max_indices = max_indices.reshape((depth, pool_height * pool_width, 2))

    for d in range(depth):
        for i in range(max_indices.shape[1]):

            # for row_index, col_index in max_indices:
            row_index = int(max_indices[d][i][0])
            col_index = int(max_indices[d][i][1])
            delta_new[d][row_index][col_index] = delta[d][i]
    return delta_new


def backprop_from_conv(deltas, weights, input_to_conv, prev_z_vals):
    '''
    Args:
     - stride
     - 
    '''
    delta_b = deltas
    # delta_w = np.dot(delta, conv_input.transpose())
    # sp = sigmoid_prime(prev_z_vals)
    # delta = np.dot(weights.transpose, delta) * sp

    delta_w = np.zeros((weights.shape))            # you need to change the dims of weights
    total_deltas_per_layer = deptas.shape[1] * deptas.shape[2]
    deltas = deltas.reshape((deltas.shape[0], deltas.shape[1] * deltas.shape[2]))

    for j in range(weights.shape[0]):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = input_to_conv[row:FILTER_SIZE+row, slide:FILTER_SIZE + slide]
            delta_w[j] += to_conv * deltas[j][i]
            slide += STRIDE

            if (FILTER_SIZE + slide)-STRIDE >= input_to_conv.shape[1]:    # wrap indices at the end of each row
                slide = 0
                row += STRIDE

    # update biases ?! return delta to conv 
    return delta_w, delta_b


def calc_gradients(delta, prev_weights, prev_activations, prev_z_vals):
    sp = sigmoid_prime(prev_z_vals)
    delta = np.dot(prev_weights.transpose(), delta) * sp                  # backprop to calculate error (delta) at layer - 1
    delta_b = delta
    delta_w = np.dot(delta, prev_activations.transpose())
    return delta_b, delta_w, delta

  
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
