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
    '''gradient of EU Loss is: (prediction - target)'''
    self.grad =  (data - target)
    return self.grad

class SigmoidCrossEntropy(object):
  """SigmoidCrossEntropy for mutli-class clasification"""
  def __init__(self, update = False):
    self.name   = "SigmoidCrossEntropy"
    self.update = update
    self.epsilon = 0.00001

  def forward(self, data, target):
    '''Problem with betting a_out=1'''
    a_out = self.sigmoid(data)
    return np.sum(np.nan_to_num(-target*np.log(data)-(1-target)*np.log(1-data)))


  def backward(self, data, target):
    a_out     = self.sigmoid(data)
    self.grad =  (a_out - target)
    return self.grad

  def sigmoid(self, data):
    a_out = 1.0/(1.0 + self.epsilon + np.exp(-data))
    return a_out

class SoftMaxLoss(object):
  """docstring for SoftMaxLoss"""
  def __init__(self,  update = False):
    self.name   = "SoftMaxLoss"
    self.update = update

  def forward(self, data, target):
    a_out = self.softmax(data)
    return np.sum(target*np.log(a_out))

  def backward(self, data, target):
    a_out = self.softmax(data)
    self.grad =  (a_out - target)
    return self.grad

  def softmax(self, data):
    a_out = np.exp(data)/np.sum(np.exp(data))
    return a_out


    