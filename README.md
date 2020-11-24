# Neural Networks and optimizers from scratch 

## Motivation
The aim of this project is to consolidate my understanding about neural networks, and to refine my internal representation of neural networks as a computation graph. 

I wanted to gain intuition about how and why different optimizers converge / behave. Therefore, I implemented 14 optimizers from scratch based on the papers they were published in. 

## Project
In this ipython notebook, I wrote a neural network with an object-oriented approach and tested it on the MNIST dataset.

For the tests, the network architecture used was 2 linear layers with relu activation followed by an output layer to a softmax function. The Layer and Model objects created can handle an arbitrary number of layers with different units.

## Optimizers
The optimizers I have implemented in this notebook includes (so far):
1. Minibatch Gradient Descent (Vanilla)
2. SGD with Momentum 
3. Nesterov Momentum (or Nesterov Accelerated Gradient)
4. Adagrad
5. Adadelta 
6. RMSprop
7. Adam 
8. Adamax
9. AdamW
10. Demon Adam 
11. Demon SGD w/ Momentum
12. Demon Nesterov Momentum
13. QHAdam
14. DemonAdamW

The graph below shows training loss over epochs for each optimizer. 
[Image]
It is noted that SGD with momentum / Nesterov momentum may be 'simple' gradient descent algorithms, but they perform quite well. With knowledge from my previous tests, these momentum optimizers are quite sensitive to the learning rate, as opposed to an algorithm from the "Adam's family".
RMSprop and Adam do perform quite well in this task, which is perhaps why they're so commonly used. 
The Adam variants show good performance in general, with QHAdam and AdamW showing promise in particular. 


## To-do
- Perhaps convert optimizers to separate objects for easier handling of arguments / optional parameters 
- Convolutional layer and pooling from scratch, to test with CIFAR10 dataset 

## References
