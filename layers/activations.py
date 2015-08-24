import math
import numpy as np

#todo: add ReLu

class Tanh(object):
  """Activation function for Neural Net"""
  def __init__(self,  update = False):
    self.a_in   = None
    self.update = update
    #assign name
    self.name   = "Tanh"
    self.idx = None
  def setIdx(self, idx):
      self.idx = idx

  def forward(self, data):
    self.a_in  = data
    self.a_out = np.tanh(data)
    return self.a_out[:]

  
  def backward(self, grad_out):
    gradient = self.gradient(self.a_out)
    back     = np.multiply(grad_out,gradient )
    return back

  # derivative of our Tanh function, in terms of the output (i.e. y)
  def gradient(self, data):
    grad = np.ones(len(data), np.float)
    grad = grad- data**2
    return grad 

import warnings
warnings.filterwarnings("error")
class Sigmoid(object):
  """Activation function for Neural Net"""
  def __init__(self,  update = False):
    self.a_in   = None
    self.update = update
    #assign name
    self.name   = "Sigmoid"
    self.idx = None

  def setIdx(self, idx):
    self.idx = idx
  
  def forward(self, data):
    self.a_in  = data
    # idx = 0
    # try:
    #   list_result = list()
    #   for d in data:
    #     a_out = 1.0/(1.0 + np.exp(-d))
    #     list_result.append(a_out)
    #     idx += 1
    # except RuntimeWarning:
    #   print 'Warning was raised as an exception!', data[idx], idx, self.name, self.idx
    self.a_out = 1.0/(1.0 + np.exp(-data))
    return self.a_out[:]

  def backward(self, grad_out):
    gradient = self.gradient(self.a_out)
    back     = np.multiply(grad_out,gradient)
    return back

  # derivative of our Sigmoid function, in terms of the output (i.e. y)
  def gradient(self, data):
    res = self.forward(data)
    grad = res*(1.0 - res)
    return grad 


