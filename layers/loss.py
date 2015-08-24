import numpy as np

#TODO: add ContrastiveLoss

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
    self.grad =  (data - target)
    return self.grad

class CrossEntropy(object):
  """SoftMaxLoss for mutli-class clasification"""
  def __init__(self, update = False):
    self.name   = "SoftMaxLoss"
    self.update = update

  def forward(self, data, target):
    return np.sum(np.nan_to_num(-target*np.log(data)-(1-target)*np.log(1-data)))


  def backward(self, data, target):
    self.grad =  (data - target)
    return self.grad