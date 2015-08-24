import numpy as np

#TODO: Add Convolutions and Pooling

class LinearNet(object):
    """Liner layer, multiply each bottom element with each top element"""
    def __init__(self, num_in, num_out, update = True, name = ''):
      self.n_in  = num_in
      self.n_out = num_out
      self.update = update
      
      #activation value
      self.a_in  = None
      self.a_out = None
      #gradient value
      self.hidden_deltas = None
      #create matrix with random value
      self.w_o = np.random.uniform(low = -1.0, size = (self.n_out, self.n_in)).astype(np.float32)  
      self.b_o = np.random.uniform(low = -1.0, high = 1.0)
      #change of weight 
      self.change       = None
      self.grad_out_acc = None #accumulated gradient
      #assign name
      self.name  = "LinerNet" + name
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
