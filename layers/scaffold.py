from activations import *
from innerproduct import *
from loss import *

''' Idea nad gradient based on http://neuralnetworksanddeeplearning.com/chap2.html'''

class NeuralNetLayer(object):
  """docstring for NeuralNetLayer"""
  def __init__(self):
    self.list_layer = list()
    self.alfa = 0.1

  
  #run forward pass throught all layers
  def forward(self, data):
    data_a = data
    for layer in self.list_layer:
      data_a = layer.forward(data_a)

    return data_a

  #run forward pass throught all layers
  def backward(self, label):
    #calculate error
    loss = self.loss_layer.forward(self.list_layer[-1].a_out, label)
    #calculate gradient
    gradient = self.loss_layer.backward(self.list_layer[-1].a_out, label)

    for layer in reversed(self.list_layer):
      gradient = layer.backward(gradient)

    return np.sum(loss)
 
  #update parameter in each layer
  def update(self):
    for layer in reversed(self.list_layer):
      if layer.update == True:
        for j in range(layer.n_in):
          for i in range(layer.n_out):
            change = layer.a_in[j] * layer.grad_out[i] #BP4
            # print layer.name, j, i, change
            layer.w_o[j,i] = layer.w_o[j,i] + self.alfa * change 
            layer.b_o = layer.b_o + self.alfa * layer.grad_out[i] 

  def train(self, data, labels,  iterations = 1000):
    for i in xrange(iterations):
      error = 0.0
      for p,l in zip(data, labels):
        self.forward(p)
        error += self.backward(l)
        self.update()
      if i % 100 == 0:
        print 'error %-14f' % error  

  def test(self, data, labels):
    for p,l in zip(data, labels):
      print p, '->', self.forward(p)

  def print_report(self):
    for idx, layer in enumerate(self.list_layer): 
      if layer.update == True:
        print idx, layer.name, layer.b_o