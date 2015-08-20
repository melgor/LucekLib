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
    self.a_out = np.zeros(len(data), np.float)
    for idx in range(len(data)):
      self.a_out[idx] = math.tanh(data[idx])
    return self.a_out[:]

  
  def backward(self, grad_out):
    back = np.zeros(len(grad_out), np.float)
    gradient = self.gradient(self.a_out)
    for idx in range(len(grad_out)):
      back[idx] = grad_out[idx] * gradient[idx]
    return back

  # derivative of our Tanh function, in terms of the output (i.e. y)
  def gradient(self, data):
    grad = np.zeros(len(data), np.float)
    for idx in range(len(data)):
      grad[idx] = 1.0 - data[idx]**2
    return grad 
