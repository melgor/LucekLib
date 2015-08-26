import random
from sklearn.metrics import accuracy_score
from activations import *
from innerproduct import *
from loss import *
import config

''' Idea and gradient based on http://neuralnetworksanddeeplearning.com/chap2.html'''

class NeuralNetLayer(object):
  """docstring for NeuralNetLayer"""
  def __init__(self):
    self.list_layer = list()
    self.list_layer_weights = list() #list of layer with parameters 

  
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
  '''  for j in range(layer.n_in):
          for i in range(layer.n_out):
            change = layer.a_in[j] * layer.grad_out[i] #BP4
            layer.w_o[j,i] = layer.w_o[j,i] + self.alfa * change 
            layer.b_o = layer.b_o + self.alfa * layer.grad_out[i] '''
  def updateDelta(self):
    for layer in reversed(self.list_layer):
      if layer.update == True:
        change = np.dot(layer.grad_out, layer.a_in.transpose())
        if layer.change is None:
          layer.change       = change
          layer.grad_out_acc = layer.grad_out
        else:
          layer.change       += change
          layer.grad_out_acc += layer.grad_out
        
  
  def updateWeighs(self):
    for layer in reversed(self.list_layer):
      if layer.update == True:
        layer.w_o          =  layer.w_o * (1 - self.lmda)- self.alfa/self.batch_size * layer.change #change weight
        # layer.w_o          -=  self.alfa/self.batch_size * layer.change #change  weight
        layer.b_o          -=  self.alfa/self.batch_size * layer.grad_out_acc #change  bias    
        layer.change       = None 
        layer.grad_out_acc = None 

  #update parameter in each layer
  def update(self):
    for layer in reversed(self.list_layer):
      if layer.update == True:
        for j in range(layer.n_in):
          for i in range(layer.n_out):
            change = layer.a_in[j] * layer.grad_out[i] #BP4
            # print layer.name, j, i, change
            layer.w_o[i,j] = layer.w_o[i,j] + self.alfa * change 
            layer.b_o = layer.b_o + self.alfa * layer.grad_out[i] 

  def train(self, data, labels,  iterations = 1000, learning_rate = 0.1):
    self.alfa = learning_rate

    for i in xrange(iterations):
      error = 0.0
      j = 0
      for p,l in zip(data, labels):
        a = np.zeros(10,np.int)
        a[l] = 1.0
        self.forward(p)
        error += self.backward(l)
        self.update()
        # print j,"error: ", error
        j += 1
      if i % 10 == 0:
        print 'error %-14f' % error  

  def sgd(self, training_data, validation_data, batch_size = 10, epochs = 30, learning_rate = 0.1, lmda = 0.1):
    '''Stochastic gradient descent algorithm '''
    #add num of Layer for each
    for idx,layer in enumerate(self.list_layer):
      layer.setIdx(idx)

    self.alfa = learning_rate
    self.lmda = lmda
    self.batch_size = float(batch_size)
    num_example = len(training_data)
    num_batch   = num_example / batch_size
    for epoch in range(epochs):
      random.shuffle(training_data)
      #create mini-batches
      mini_batches =  [training_data[k:k+batch_size] for k in xrange(0, num_example, batch_size)]
      error = 0.0
      # j = 0
      config.phase = "Train"
      for batch in mini_batches:
        for x,y in batch:
          self.forward(x)
          error += self.backward(y)
          self.updateDelta()
        self.updateWeighs()
      #test data
      config.phase = "Test"
      list_result = list()
      list_groud = list()
      for x,y in validation_data:
        list_result.append(np.argmax(self.forward(x)))
        list_groud.append(y)
      print "Epoch: ", epoch, " ACC: ", accuracy_score(np.asarray(list_groud),np.asarray(list_result))


  def test(self, data, labels):
    for p,l in zip(data, labels):
      print p, '->', self.forward(p)

  def print_report(self):
    for idx, layer in enumerate(self.list_layer): 
      if layer.update == True:
        print idx, layer.name, layer.b_o