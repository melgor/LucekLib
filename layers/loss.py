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
  """SoftMaxLoss for mutli-class clasification with probability interpretation"""
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

class ContrastiveLoss(object):
  """ContrastiveLoss for learning embedding"""
  def __init__(self, update = False, margin = 1.0):
    self.name    = "ContrastiveLoss"
    self.update  = update
    self.margin  = margin
    self.epsilon = 0.0001


  def forward(self, data, target):
    '''Data must be a tuple of features to compare, \
       target show if corresponding features represent same/not same class\
       Loss = 0.5 * Y * ||D1 -D2||**2 + 0.5 * (1 - Y)  * max(0, margin -||D1 -D2|| ) ** 2
       '''

    self.data_1     = data[0]
    self.data_2     = data[1]
    if target == 1: #same
      self.error      = (self.data_1 -  self.data_2)
      self.error_pow2 = np.sum(self.error **2) 
    else: #not same
      self.diff       = self.data_1 -  self.data_2
      self.dist       = np.sqrt(np.sum(self.diff**2))
      self.error      = self.margin - self.dist 
      self.error_pow2 = np.maximum(0.0, self.error ) ** 2
    
    loss            = (error_pow2)/(2.0)
    return loss


  def backward(self, data, target):
    '''gradient: 
       positive: D1 -D2 (in case of D1) or -(D1 - D2) (in case of D2)
       negative: if (margin -(D1 -D2)) > 0, then (margin - ||D1 - D2||)/||D1 - D2|| *(D1 - D2) * (-1)  (in case of D1);
                                                 (margin - ||D1 - D2||)/||D1 - D2|| *(D1 - D2) * (1)   (in case of D2)
                 else: 0 

    '''
    self.data_1 = data[0]
    self.data_2 = data[1]
    if target == 1: #same
      grad_d1   =   self.error 
      grad_d2   = - self.error 
    else: #not same
      if self.error > 0.0:
       grad_d1   = - self.error/(self.dist + self.epsilon) * self.diff 
       grad_d2   = - self.grad_d1 
      else:
        grad_d1 = 0.0
        grad_d2 = 0.0
  
    return (grad_d1, grad_d2)









    