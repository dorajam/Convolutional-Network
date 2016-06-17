# Convolutional network
###### - an implementation of a deep convolutional neural network -    
###### - done in pure Python without using any machine learning frameworks -    



The purpose of this project was to understand the full architecture of a conv net and to visually break down what's going on while training to recognize images.

#### DONE:

###### Setting up:    
- set up the architecutre by specifying the number of layers (convolutional, pooling, fully connected (FC) and final (classify) layer)
- make sure to use an FC layer as your last layer before classifying
- play around with your hyperparameters      
 

###### Training:    
- use ```run.py``` to start the training
- see the output images on each convolutional layer + pooling layer while training
- see the filters being trained ! They should slowly resemble the festures in your images


#### TO IMPROVE:
Even though you can get some insights of the learning that's happening during the training, the network is extremely slow!
This is mainly because it was never designed and optimized to process large volume of images.
It would be great to rewrite this in Theano on Tensorflow


#### FURTHER IDEAS:
- once optimized -> build an interactive visualization in the browser of the filters being trained
- building a generative model that reproduces the image based on a label



