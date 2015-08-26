'''Own impelmentation of Neural Nets

  Idea is to learn implementation to create TripletLoss to reproduce FaceNet/Baidu results
  TripletLoss need non standard operation: collecting triplets (anchor, neg, pos), what can be done 
  online or offline.
'''

import numpy as np
import math
import random
import cPickle, gzip
from layers.scaffold import *
from layers.network2 import *


random.seed(110)
np.random.seed(110)

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

def load_data():
  f = gzip.open('data/mnist.pkl.gz', 'rb')
  training_data, validation_data, test_data = cPickle.load(f)
  f.close()
  return (training_data, validation_data, test_data)

def load_data_wrapper():
  tr_d, va_d, te_d = load_data()
  training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = zip(training_inputs, training_results)
  validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
  validation_data = zip(validation_inputs, va_d[1])
  test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
  test_data = zip(test_inputs, te_d[1])
  return (training_data, validation_data, test_data)

def demoMnist():
  training_data, validation_data, test_data = load_data_wrapper()
  n = NeuralNetLayer()
  n.list_layer.append(LinearNet(784,100))
  n.list_layer.append(ReLU())
  n.list_layer.append(LinearNet(100,10))
  # n.list_layer.append(DropOut())
  # n.list_layer.append(Sigmoid())
  n.loss_layer = SigmoidCrossEntropy() #SigmoidCrossEntropy() # EuclidesianLoss() #SoftMaxLoss()

  n.sgd(training_data, validation_data, learning_rate = 0.5, lmda = 0.00005)

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
    n.loss_layer = CrossEntropy()
    n.train(data, labels, iterations = 10000, learning_rate = 3.0)
    n.test(data, labels)
    # print n.print_report()
    


if __name__ == '__main__':
    # demoClassification()
    demoMnist()
