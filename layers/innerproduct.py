import numpy as np

class LinearNet(object):
    """Liner layer, multiply each bottom element with each top element"""
    def __init__(self, num_in, num_out, update = True):
      self.n_in  = num_in
      self.n_out = num_out
      self.update = update
      
      #activation value
      self.a_in  = np.ones(self.n_in, np.float)
      self.a_out = np.ones(self.n_out, np.float)
      #gradient value
      self.hidden_deltas = [0.0] * self.n_out
      #create matrix with random value
      self.w_o = np.random.uniform(low = -1.0, size = (self.n_in,self.n_out))  
      self.b_o = np.random.uniform(low = -1.0, high = 1.0)
      #assign name
      self.name = "LinerNet"


    def forward(self, data):
      self.a_in  = data
      #calculate activation to hidden layer
      # print "Size: ", data.shape, self.w_o.shape, np.dot(self.w_o, self.a_in)
      # for j in range(self.n_out):
      #   total = 0.0
      #   for i in range(self.n_in):
      #     total += self.a_in[i] * self.w_o[i,j]
      #   self.a_out[j] = total + self.b_o
      self.a_out = np.dot(self.a_in, self.w_o,) + self.b_o
      return self.a_out[:]

    def backward(self, grad_out):
      #calculate error for hidden layer BP2
      self.grad_out      = grad_out
      self.hidden_deltas = [0.0] * self.n_in
      for j in range(self.n_in):
        error = 0.0
        for i in range(self.n_out):
          error += grad_out[i] * self.w_o[j, i] 
        self.hidden_deltas[j] = error
      
      return self.hidden_deltas[:]
