import math
import numpy as np

class Tanh(object):
  """Activation function for Neural Net"""
  def __init__(self,  update = False):
    self.a_in = None
    self.update = update
    #assign name
    self.name = "Tanh"

  def forward(self, data):
    self.a_in = data
    self.a_out = np.tanh(data)
    return self.a_out[:]

  
  def backward(self, grad_out):
    gradient = self.gradient(self.a_out)
    back = np.multiply(grad_out,gradient )
    return back

  # derivative of our Tanh function, in terms of the output (i.e. y)
  def gradient(self, data):
    grad = np.ones(len(data), np.float)
    grad = grad- data**2
    return grad 
