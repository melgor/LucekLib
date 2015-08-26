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
    gradient = self.gradient(self.a_in)
    back     = np.multiply(grad_out,gradient )
    return back

  # derivative of our Tanh function, in terms of the intput (i.e. x)
  def gradient(self, data):
    grad = np.ones(len(data), np.float)
    grad = grad- data**2
    return grad 

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
    self.a_out = 1.0/(1.0 + np.exp(-data))
    return self.a_out[:]

  def backward(self, grad_out):
    gradient = self.gradient(self.a_in)
    back     = np.multiply(grad_out,gradient)
    return back

  # derivative of our Sigmoid function, in terms of the intput (i.e. x)
  def gradient(self, data):
    grad = self.a_out*(1.0 - self.a_out)
    return grad 

class SoftMax(object):
  """docstring for SoftMax"""
  @staticmethod
  def forward(data):
    return np.exp(data)/np.sum(np.exp(data))

class ReLU(object):
  """docstring for ReLU"""
  def __init__(self,  update = False):
    self.a_in   = None
    self.update = update
    #assign name
    self.name   = "ReLu"
    self.idx = None

  def setIdx(self, idx):
      self.idx = idx
      
  def forward(self, data):
    self.a_in  = data
    self.a_out = np.where(data > 0, data, 0)
    return self.a_out[:]

  def backward(self, grad_out):
    back = np.where(self.a_in > 0, grad_out, 0)
    return back
 
    
 
    


