# Neural Networks and optimizers from scratch 

## Motivation
The aim of this project is to consolidate my understanding about neural networks, and to refine my internal representation of neural networks as a computation graph. 

I wanted to gain intuition about how and why different optimizers converge / behave. Therefore, I implemented a number of optimizers from scratch based on the papers they were published in. 

## Project
In [this ipython notebook](https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/Neural%20Networks%20and%20Optimizers%20from%20scratch%20in%20NumPy.ipynb), I wrote a neural network with an object-oriented approach and tested it on the MNIST dataset. The optimisers are contained in [this script](https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/optimizers.py).

For the tests, the network architecture used was 2 linear layers with relu activation followed by an output layer to a softmax function. The Layer and Model objects created can handle an arbitrary number of layers with different units.

## Optimizers
The optimizers I have implemented in this notebook includes (so far):
1. Minibatch Gradient Descent (Vanilla)
2. SGD with Momentum 
3. [Nesterov Momentum](https://arxiv.org/pdf/1212.0901v2.pdf) (or Nesterov Accelerated Gradient)
4. [Adagrad](https://jmself.learning_rate.org/papers/volume12/duchi11a/duchi11a.pdf)
5. [RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
6. [Adam](https://arxiv.org/pdf/1412.6980.pdf)
8. [Nadam](http://cs229.stanford.edu/proj2015/054_report.pdf)
9. [Adadelta](https://arxiv.org/pdf/1212.5701.pdf) 
10. [Adamax](https://arxiv.org/pdf/1412.6980.pdf)
13. [QHAdam](https://arxiv.org/pdf/1810.06801.pdf)

[Decaying Momentum (Demon)](https://arxiv.org/pdf/1910.04952v3.pdf) can be applied to any optimizer that inherits from the Adam subclass and the SGDM subclass, and [Decoupled Weight decay](https://arxiv.org/pdf/1711.05101v3.pdf) can be applied to any optimizer that inheritis from the Adam subclass. This can result in optimizers such as DemonQHAdamW or DemonNesterov.


The graph below shows training loss over epochs for a few select optimizers: 
![img](https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/media/Loss.png)

This one shows validation accuracy over epochs:
![img](https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/media/Valacc.png)

QHAdamW performed the best in training loss, while Nesterov performed the best in validation accuracy in this task.

It is noted that SGD with momentum / Nesterov momentum may be 'simpler' gradient descent algorithms, but they perform quite well over in convergence over epochs.

With knowledge from my previous tests, these momentum optimizers are quite sensitive to the learning rate, as opposed to an algorithm from the "Adam's family".



## To-do
-  ~~Perhaps convert optimizers to separate objects for easier handling of arguments / optional parameters~~ 
- Convolutional layer and pooling from scratch, to test with CIFAR10 dataset 
