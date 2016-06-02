# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''Test for nneuwork.py for MNIST dataset, accuracy will change with batch size/number of epochs. See additional files'''

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import nnetwork
# input layers are for the MNIST dataset where each image is of 28 x 28
net = nnetwork.Network([784, 30, 10])
# the arguments are the following: training data, batch size, learning rate and number of epochs
net.gradientDescent(training_data, 10, 2.5, 30, test_data=test_data)


