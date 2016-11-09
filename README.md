# Convolutional network
- _an implementation of a deep convolutional neural network_     
- _done in Python and Numpy, with no external machine learning framework used_    



The purpose of this project was to understand the full architecture of a conv net and to visually break down what's going on while training to recognize images. In particular, I was interested in seeing how the weight kernels pick up some pattern over the course of the training.

#### DONE:

###### Setting up:    
- [x] set up the architecutre by specifying the number of layers (convolutional, pooling, fully connected (FC) and final (classify) layer)
- [x] make sure to use an FC layer as your last layer before classifying
- [x] tweak your hyperparameters      
- [x] make sure you see the pattern changes in the weight kernels after each batch update      
 

###### Training:    
- use ```run.py``` to start the training
- see the output images on each convolutional layer + pooling layer while training
- see the filters being trained ! They should slowly resemble the feqtures in your images


#### TO IMPROVE:
Even though you can get some insights into the learning during training, the network is extremely slow!
This is mainly because it was never designed and optimized to process large volume of images.
It would be great to rewrite this in Theano or Tensorflow


#### FURTHER IDEAS:
- once optimized -> build an interactive visualization in the browser of the filters being trained
- building a generative model that reproduces the image based on a label
- maximizing a given classification (final activation) by backpropagating all the way to your image (input layer) and updating pixel values -> could provide some better intuition on what the network thinks of a class's features.



