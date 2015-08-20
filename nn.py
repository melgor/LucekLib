'''Own impelmentation of Neural Nets

  Idea is to learn implementation to create TripletLoss to reproduce FaceNet/Baidu results
  TripletLoss need non standard operation: collecting triplets (anchor, neg, pos), what can be done 
  online or offline.
'''

import numpy as np
import math
import random
from layers.scaffold import *


random.seed(0)
np.random.seed(0)


def demoMnist():
  data = np.load('mnist_data.npy')
  target = np.load('mnist_labels.npy')
  n = NeuralNetLayer()
  n.list_layer.append(LinearNet(784,100))
  n.list_layer.append(Tanh())
  n.list_layer.append(LinearNet(100,1))
  n.list_layer.append(Tanh())
  n.loss_layer = EuclidesianLoss()


def demoClassification():
    # Teach network XOR function
    data = np.zeros((4,2), np.float )
    data[1,1] = 1.0
    data[2,0] = 1.0
    data[3,0] = 1.0
    data[3,1] = 1.0

    labels = np.array([[0], [1],[1], [0]], np.int)
    # create a network with two input, two hidden, and one output nodes
    n = NeuralNetLayer()
    n.list_layer.append(LinearNet(2,10))
    n.list_layer.append(Tanh())
    # n.list_layer.append(LinearNet(5,5))
    # n.list_layer.append(Tanh())
    n.list_layer.append(LinearNet(10,1))
    n.list_layer.append(Tanh())
    n.loss_layer = EuclidesianLoss()
    n.train(data, labels, iterations = 1000)
    n.test(data, labels)
    # print n.print_report()
    


if __name__ == '__main__':
    #demoRegression()
    demoClassification()
