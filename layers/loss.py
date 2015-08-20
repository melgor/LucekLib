import numpy as np


class EuclidesianLoss(object):
  """EuclidesianLoss which is defined as = 0.5(target -  activation)**2"""
  def __init__(self, update = False):
    self.a_in   = None
    self.loss   = None
    self.update = update
    self.name = "EuclidesianLoss"

  def forward(self, data, target):
    self.loss =   0.5 *(target -  data)
    return self.loss
  
  def backward(self, data, target):
    '''gradient of EU Loss is: (perdiction - target)'''
    self.grad =   -(data - target)#change direction of gradient
    return self.grad