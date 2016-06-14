# Dora Jambor
# May 2016
# helper functions for backpropagation

import numpy as np


# backpropagation
##############################################################
# I. Go from last layer to FC layer
def backprop_final_to_fc(prev_activation, z_vals, final_output, y):
    # print 'prev_act: ', prev_activation.shape,'z_vals: ', z_vals.shape
    delta = (final_output - y) * sigmoid_prime(z_vals)
    delta_b = delta
    delta_w = np.dot(delta, prev_activation.transpose())
    print "delta_w shape in FC to inp layer:", delta_w.shape
    return delta_b, delta_w, delta

def backprop_fc_to_pool(delta, prev_weights, prev_activations, z_vals):
    x,y,z = prev_activations.shape
    prev_activations = prev_activations.reshape((x*y*z, 1))
    return calc_gradients(delta, prev_weights, prev_activations, z_vals)
    
def backprop_pool_to_conv(delta, prev_weights, input_from_conv, max_indices, poolsize, pool_output):
    # reshape the "z values" of the pool layer
    x,y,z = pool_output.shape
    pool_output= pool_output.reshape((x * y * z,1))

    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))

    # backprop delta from fc to pool layer
    sp = sigmoid_prime(pool_output)
    delta = np.dot(prev_weights.transpose(), delta) * sp         # backprop to calc delta on pooling layer
    delta = delta.reshape((x,y*z))
    pool_output= pool_output.reshape((x, y * z))

    print 'my delta on fc before pooling:' ,delta.shape

    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width)) # calc the delta on the conv layer

    for d in range(depth):    # depth is the same for conv + pool layer
        row = 0
        slide = 0
        for i in range(max_indices.shape[1]):
            toPool = input_from_conv[d][row:poolsize[0] + row, slide:poolsize[0] + slide]

            # calculate the new delta for the conv layer based on the max result + pooling input
            delta_new[d][row:poolsize[0] + row, slide:poolsize[0] + slide] = max_prime(pool_output[d][i], delta[d][i], toPool)

            slide += poolsize[1]
            if slide >= width:
                slide = 0
                row+= poolsize[1]

    return delta_new

def backprop_conv_to_conv(delta, weight_filters, stride, input_to_conv, prev_z_vals):
    '''
    Args:
     - stride
     - 
    '''
    # this is 4 dims: num_filters, depth, height, width
    # print 'filter dimensions: ', weight_filters.shape
    num_filters, depth, filter_size, filter_size = weight_filters.shape

    # print 'conv input shape:', input_to_conv.shape

    delta_b = np.zeros((weight_filters.shape[0], 1))
    delta_w = np.zeros((weight_filters.shape))            # you need to change the dims of weights
    total_deltas_per_layer = delta.shape[1] * delta.shape[2]
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))
    new_delta = np.zeros((input_to_conv.shape))

    for j in range(num_filters):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = input_to_conv[:, row:filter_size+row, slide:filter_size + slide]
            delta_w[j] += to_conv * delta[j][i]
            delta_w[j] += delta[j][i]       # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (filter_size + slide)-stride >= input_to_conv.shape[2]:    # wrap indices at the end of each row
                slide = 0
                row += stride
    return delta_b, delta_w

def calc_gradients(delta, prev_weights, prev_activations, z_vals):
    sp = sigmoid_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(prev_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    dim1, dim2 = prev_activations.shape
    prev_activations = prev_activations.reshape((1,dim1*dim2))
    delta_w = np.dot(delta, prev_activations)
    return delta_b, delta_w, delta

def backprop_fc_to_input(delta, prev_weights, prev_activations, z_vals):
    sp = sigmoid_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(prev_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    depth, dim1, dim2 = prev_activations.shape
    prev_activations = prev_activations.reshape((1,depth*dim1*dim2))
    delta_w = np.dot(delta, prev_activations)
    return delta_b, delta_w, delta


def max_prime(res, delta, tile_to_pool):
    dim1, dim2 = tile_to_pool.shape
    tile_to_pool = tile_to_pool.reshape((dim1 * dim2))
    new_delta = np.zeros((tile_to_pool.shape))
    for i in range(len(tile_to_pool)):
        num = tile_to_pool[i]
        if num < res:
            new_delta[i] = 0
        else:
            new_delta[i] = delta
    return new_delta.reshape((dim1, dim2))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
