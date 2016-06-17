# Dora Jambor
# May 2016
# helper functions for backpropagation

import numpy as np


# backpropagation
##############################################################

def backprop_1d_to_1d(delta, prev_weights, prev_activations, z_vals, final=False):
    if not final: # reset delta
        sp = sigmoid_prime(z_vals)
        # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
        delta = np.dot(prev_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    delta_w = np.dot(delta, prev_activations.transpose())
    return delta_b, delta_w, delta

def backprop_1d_to_3d(delta, prev_weights, prev_activations, z_vals):
    sp = sigmoid_prime(z_vals)
    # print 'w,d,z_vals: ', prev_weights.shape, delta.shape, sp.shape, prev_activations.shape
    delta = np.dot(prev_weights.transpose(), delta) * sp         # backprop to calculate error (delta) at layer - 1

    delta_b = delta
    depth, dim1, dim2 = prev_activations.shape
    prev_activations = prev_activations.reshape((1, depth*dim1*dim2))
    delta_w = np.dot(delta, prev_activations)
    delta_w = delta_w.reshape((delta.shape[0], depth,dim1,dim2))

    return delta_b, delta_w, delta

    
def backprop_pool_to_conv(delta, prev_weights, input_from_conv, max_indices, poolsize, pool_output):
    # reshape the "z values" of the pool layer
    x,y,z = pool_output.shape
    a,b,c,d = prev_weights.shape
    prev_weights = prev_weights.reshape((a,b*c*d))
    pool_output= pool_output.reshape((x * y * z,1))

    # same for the max index matrix
    max_indices = max_indices.reshape((x, y * z, 2))

    # backprop delta from fc to pool layer
    sp = sigmoid_prime(pool_output)
    delta = np.dot(prev_weights.transpose(), delta) * sp         # backprop to calc delta on pooling layer
    delta = delta.reshape((x,y*z))
    pool_output = pool_output.reshape((x, y * z))
    
    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width)) # calc the delta on the conv layer

    for d in range(depth):    # depth is the same for conv + pool layer
        row = 0
        slide = 0
        for i in range(max_indices.shape[1]):
            toPool = input_from_conv[d][row:poolsize[0] + row, slide:poolsize[0] + slide]

            # calculate the new delta for the conv layer based on the max result + pooling input
            deltas_from_pooling = max_prime(pool_output[d][i], delta[d][i], toPool)
            delta_new[d][row:poolsize[0] + row, slide:poolsize[0] + slide] = deltas_from_pooling

            slide += poolsize[1]
            if slide >= width:
                slide = 0
                row+= poolsize[1]

    return delta_new

def backprop_to_conv(delta, weight_filters, stride, input_to_conv, prev_z_vals):
    '''weights passed in are the ones between pooling and fc layer'''

    # print 'weight filter, delta shape', weight_filters.shape, delta.shape
    # print 'input shape', input_to_conv.shape
    num_filters, depth, filter_size, filter_size = weight_filters.shape

    delta_b = np.zeros((num_filters, 1))
    delta_w = np.zeros((weight_filters.shape))            # you need to change the dims of weights

    # print delta_w.shape, delta_b.shape, delta.shape
    total_deltas_per_layer = (delta.shape[1]) * (delta.shape[2])
    # print 'total_deltas_per_layer', total_deltas_per_layer
    delta = delta.reshape((delta.shape[0], delta.shape[1] * delta.shape[2]))

    for j in range(num_filters):
        slide = 0
        row = 0

        for i in range(total_deltas_per_layer):
            to_conv = input_to_conv[:,row:filter_size+row, slide:filter_size + slide]
            delta_w[j] += to_conv * delta[j][i]
            delta_b[j] += delta[j][i]       # not fully sure, but im just summing up the bias deltas over the conv layer
            slide += stride

            if (slide + filter_size)-stride >= input_to_conv.shape[2]:    # wrap indices at the end of each row
                slide = 0
                row+=stride

    return delta_b, delta_w


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
