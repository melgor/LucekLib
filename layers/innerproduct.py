import numpy as np

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
      self.w_o = np.random.uniform(low = -1.0, size = (self.n_in,self.n_out))  
      self.b_o = np.random.uniform(low = -1.0, high = 1.0)
      #assign name
      self.name = "LinerNet"


    def forward(self, data):
      self.a_in  = data
      #calculate activation to hidden layer
      self.a_out = np.dot(self.a_in, self.w_o) + self.b_o
      return self.a_out[:]

    def backward(self, grad_out):
      #calculate error for hidden layer BP2
      # for j in range(self.n_in):
      #   error = 0.0
      #   for i in range(self.n_out):
      #     error += grad_out[i] * self.w_o[j, i] 
      #   self.hidden_deltas[j] = error
      self.grad_out      = grad_out
      self.hidden_deltas = np.dot(self.w_o, grad_out)
      return self.hidden_deltas[:]
