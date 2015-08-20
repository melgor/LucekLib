import numpy as np


class EuclidesianLoss(object):
  """EuclidesianLoss which is defined as = 0.5(target -  activation)**2"""
  def __init__(self, update = False):
    self.a_in   = None
    self.loss   = None
    self.update = update
    self.name = "EuclidesianLoss"

  def forward(self, data, target):
    self.loss = np.zeros(len(target), np.float) 
    for idx in range(len(target)):
      self.loss[idx] = 0.5 *(target[idx] -  data[idx])**2
    return self.loss
  
  def backward(self, data, target):
    '''gradient of EU Loss is: (perdiction - target)'''
    self.grad = [0.0] * len(target) 
    for idx in range(len(target)):
      self.grad[idx] = -(data[idx] - target[idx]) #change direction of gradient
    return self.grad