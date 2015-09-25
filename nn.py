'''Own impelmentation of Neural Nets

  Idea is to learn implementation to create TripletLoss to reproduce FaceNet/Baidu results
  TripletLoss need non standard operation: collecting triplets (anchor, neg, pos), what can be done 
  online or offline.
'''

import numpy as np
import math
import random
from layers.scaffold import *
from layers.network2 import *
from data.data_load import *

random.seed(110)
np.random.seed(110)


def demoMnist():
  batch_size = 10
  training_data, validation_data, test_data = load_data_wrapper()
  list_ver_1, list_ver_2, target = create_verification_task(test_data, batch_size)
  n = NeuralNetLayer()
  n.list_layer.append(LinearNet(784,100))
  n.list_layer.append(ReLU())
  n.list_layer.append(LinearNet(100,10))
  # n.list_layer.append(DropOut())
  # # n.list_layer.append(Sigmoid())
  n.loss_layer = ContrastiveLoss(margin = 1.0) #SigmoidCrossEntropy() # EuclidesianLoss() #SoftMaxLoss() #ContrastiveLoss(margin = 1.0)

  # n.sgd(training_data, validation_data, learning_rate = 0.5, lmda = 0.00005)

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
