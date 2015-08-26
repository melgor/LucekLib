import numpy as np
import config
import copy
#TODO: Add Convolutions and Pooling

class LinearNet(object):
    """Liner layer, multiply each bottom element with each top element"""
    def __init__(self, num_in, num_out, update = True):
      self.n_in  = num_in
      self.n_out = num_out
      self.update = update
      
      #activation value
      self.a_in  = None
      self.a_out = None
      #gradient value
      self.hidden_deltas = None
      #create matrix with random value
      self.w_o = np.random.randn(self.n_out, self.n_in) /np.sqrt(self.n_in)
      self.b_o = np.random.randn(self.n_out, 1)
      #change of weight 
      self.change       = None
      self.grad_out_acc = None #accumulated gradient
      #assign name
      self.name  = "LinerNet"
      self.idx   = None
      self.debug = False
    
    def setIdx(self, idx):
      self.idx = idx

    def forward(self, data):
      self.a_in  = data
      #calculate activation to hidden layer
      self.a_out = np.dot(self.w_o, self.a_in) + self.b_o
      return self.a_out[:]

    def backward(self, grad_out):
      #calculate error for hidden layer BP2
      # self.hidden_deltas = np.zeros((self.n_in,1))
      # for j in range(self.n_in):
      #   error = 0.0
      #   for i in range(self.n_out):
      #     error += grad_out[i] * self.w_o[i, j] 
      #   self.hidden_deltas[j] = error

      self.grad_out      = grad_out
      self.hidden_deltas = np.dot( self.w_o.transpose(), grad_out)
      return self.hidden_deltas[:]

    def weightInit(self, type_init = "gaussian"):
      self.b_o = np.random.randn(self.n_out, 1)
      if type_init == "gaussian":
        self.w_o = np.random.randn(self.n_out, self.n_in)
      elif type_init == "msra":
        self.w_o = np.random.randn(self.n_out, self.n_in) /np.sqrt(2.0/self.n_in)
      elif type_init == "xavier":
        self.w_o = np.random.randn(self.n_out, self.n_in) /np.sqrt(self.n_in)



class DropOut(object):
  """docstring for DropOut"""
  def __init__(self, p = 0.5, update = False):
    self.p = 0.5
    self.update = False

  def setIdx(self, idx):
    self.idx = idx

  '''Inverted DropOut, idea taken from http://cs231n.github.io/neural-networks-2/'''
  def forward(self, data):
    self.a_in  = copy.copy(data)
    if config.phase == "Train":
      self.mask = (np.random.rand(*data.shape) < self.p) / self.p # first dropout mask scaled 
      data *= self.mask # drop!
    self.a_out = data
    return self.a_out

  def backward(self, grad_out):
    '''Gradient of dropout based on input is: mask / p, so it is self.mask '''
    self.grad_out = grad_out
    return grad_out * self.mask

    