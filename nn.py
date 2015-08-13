'''Own impelmentation of Neural Nets

  Idea is to learn implementation to create TripletLoss to reproduce FaceNet/Baidu results
  TripletLoss need non standard operation: collecting triplets (anchor, neg, pos), what can be done 
  online or offline.
'''

import numpy as np
import math
import random

random.seed(0)

    
# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NeuralNet(object):
  """docstring for NeuralNet"""
  def __init__(self, n_in, n_hidden, n_out):
    self.n_in     = n_in + 0    #bias
    self.n_hidden = n_hidden + 0 #bias
    self.n_out    = n_out

    # activations for nodes
    self.a_in     = [1.0]*self.n_in
    self.a_hidden = [1.0]*self.n_hidden
    self.a_out    = [1.0]*self.n_out

    #init weights
    self.w_i = np.random.uniform(low = -1.0, size = (self.n_in, self.n_hidden))
    self.w_o = np.random.uniform(low = -1.0, size = (self.n_hidden,self.n_out))

    self.alfa = 0.1 #learning rate

  #forward propagation of neural net
  def forward(self, input_data):
    #remember input data
    self.a_in = input_data

    #calculate activation to hidden layer
    for j in range(self.n_hidden):
      total = 0.0
      for i in range(self.n_in):
        total += self.a_in[i] * self.w_i[i,j]
      self.a_hidden[j] = sigmoid(total)
  
    #calculate activation to output layer
    for j in range(self.n_out):
      total = 0.0
      for i in range(self.n_hidden):
        total += self.a_hidden[i] * self.w_o[i,j]
      self.a_out[j] = sigmoid(total)


    return self.a_out[:]

  #backward propagation of net
  def backward(self, label):
    
    # calculate error terms for output BP1
    output_deltas = [0.0] * self.n_out
    for j in range(self.n_out):
      output_deltas[j] = label[0] - self.a_out[j] #gradient of loss function
      output_deltas[j] = dsigmoid(self.a_out[j]) * output_deltas[j] #gradient of pre-acrivation function

    #calculate error for hidden layyer BP2
    hidden_deltas = [0.0] * self.n_hidden
    for j in range(self.n_hidden):
      error = 0.0
      for i in range(self.n_out):
        error += output_deltas[i] * self.w_o[j, i] 
      hidden_deltas[j] = dsigmoid(self.a_hidden[j]) * error
    
    # update output weights
    for j in range(self.n_hidden):
      for i in range(self.n_out):
        change = output_deltas[i] * self.a_hidden[j]
        self.w_o[j,i] = self.w_o[j,i] + self.alfa * change

    # update input weights
    for j in range(self.n_in):
      for i in range(self.n_hidden):
        change = hidden_deltas[i] * self.a_in[j]
        self.w_i[j, i] = self.w_i[j, i] + self.alfa * change

    # calculate error
    error = 0.5 * (label[0] - self.a_out[0])**2
    return error

  def train(self, data, iterations = 1000):
    for i in xrange(iterations):
      error = 0.0
      for p in data:
        self.forward(p[0])
        error += self.backward(p[1])
      if i % 100 == 0:
        print 'error %-14f' % error

  def test(self, data, verbose = False):
        tmp = []
        for p in data:
            if verbose:
                print p[0], '->', self.forward(p[0])
            tmp.append(self.forward(p[0]))

        return tmp

def demoClassification():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NeuralNet(2, 5, 1)
    n.train(pat)
    n.test(pat, verbose = True)
    


if __name__ == '__main__':
    #demoRegression()
    demoClassification()
