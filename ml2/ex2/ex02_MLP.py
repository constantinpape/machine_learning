"""
Created on Sun Nov 30 11:22:29 2014

@author: Marius Felix Killinger
"""

import numpy as np
from sklearn import datasets

class Node(object):
  def __init__(self):
    self.input   = None   
  def forward(self, input):
    return input    
  def backward(self, top_error):
    return top_error

class Tanh(Node):
  def __init__(self):
    self.input   = None    
  def forward(self, input):
    self.input = input
    return ...
  def backward(self, top_error):
    """ return d out / d in """ 
    return ...


class PerceptronLayer(Node):
  def __init__(self, nin, nout):
    self.nin  = nin
    self.nout = nout
    self.W    = np.random.uniform(low=-np.sqrt(6. / (nin + nout)),
                           high=np.sqrt(6. / (nin + nout)), size=(nin, nout)).astype(np.float32)
    self.b    = np.random.uniform(-1e-8,1e-8,(nout,)).astype(np.float32)
   
    self.lin  = None   # stores the dot product of w and the last input
    self.act  = None   # stores the mapping of lin with the activation function
    self.input  = None # stores the last input
    self.grad_b = None
    self.grad_W = None
    self.act_func = Tanh()

          
  def forward(self, input):
    """ (bs, n_in) --> (bs, n_out) """
    self.lin =   ...
    self.act =   ...
    self.input = ...
    return self.act   

  def backward(self, top_error):
    """ d out / d in """ 
    self.act_error = ...
    err = ...
    return err
  
  def grad(self, top_error):
    """ d out / d W """ 
    grad_b = ...
    grad_W = ...
    self.grad_b = grad_b
    self.grad_W = grad_W
    return grad_W, grad_b
    
  def GD_update(self, lr):
    self.W = self.W - (lr*self.grad_W)
    self.b = self.b - (lr*self.grad_b)

    
class Softmax(Node):
  def __init__(self):
    self.input   = None
    
  def forward(self, input):
    self.input = input
    """ return softmax function to input vector"""
    return ...
    
  def backward(self, top_error):
  	""" return the back propagation error """
    return ...


class NLL(object):
  def __init__(self, n_lab):
    self.input   = None
    self.classes = np.arange(n_lab, dtype=np.int)[None,:]
    
  def forward(self, input, Y):
    self.input = input
    self.n     = Y.shape[0]
    self.active_class = np.equal(self.classes,Y[:,None])
    return ...
    
  def backward(self):
    return ...


class MLP(object):
  def __init__(self, layer_sizes, nin):
    self.layers     = []
    self.last_grads = None
    n_lay           = len(layer_sizes)
    for i in xrange(n_lay-1):
      print "Adding layer (#in %i, #out %i)" % (nin, layer_sizes[i])
      self.layers.append(PerceptronLayer(nin, layer_sizes[i]))
      nin = layer_sizes[i]
      
    print "Adding layer (#in %i, #out %i)" % (nin, layer_sizes[-1])  
    self.layers.append(PerceptronLayer(nin,layer_sizes[-1], act_func='lin'))
    
    self.softmax = Softmax()
    self.loss = NLL(layer_sizes[-1])
    
  def forward(self, X):
    bs = X.shape[0]
    X = X.reshape(bs, -1)
    result = X
    for lay in self.layers:
      result = lay.forward(result)
    return result
    
  def class_prob(self, X):
    out = self.forward(X) 
    return self.softmax.forward(out)
     
  def get_loss(self, X, Y):
    pred = self.class_prob(X)
    loss = self.loss.forward(pred, Y)
    cls  = pred.argmax(axis=1)
    acc  = 1-np.mean(np.equal(cls, Y))
    return loss, acc
    
    
  def gradients(self, X, Y):
    class_prob = self.class_prob(X)
    loss    = self.loss.forward(class_prob, Y)
    
    top_err = self.loss.backward()
    top_err = self.softmax.backward(top_err)
    grads   = []
    for lay in self.layers[::-1]:
      new_err = lay.backward(top_err)
      grad_W, grad_b = lay.grad(top_err)
      grads.append(grad_b)
      grads.append(grad_W)
      top_err = new_err
    self.last_grads = grads
    return grads[::-1], loss, class_prob
    
  def update(self, lr):
    for lay in self.layers:
      lay.GD_update(lr)



    
if __name__=="__main__":

  n = 2000

  X, Y = datasets.make_moons(n, noise=0.05)
  X_test   = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
  X_test   = np.vstack((X_test[0].flatten(), X_test[1].flatten())).T
  X -= X.min(axis=0)
  X /= X.max(axis=0)
  X  = (X - 0.5) * 2
  
  nin = 2
  bs  = 200
  lr  = 0.05
  
  trace = dict(X=X, Y=Y, W1=[], b1=[], a1=[], W2=[], b2=[], a2=[], dec=[])
  nn = MLP([3,2], nin)
  
  pos = 0
  perm = np.random.permutation(n)
  
  nn = MLP([3,3,10], nin)
  for i in xrange(10000):
    grads, loss, pred = nn.gradients(X[perm[pos:pos+bs]], Y[perm[pos:pos+bs]])
    nn.update(lr)
    if i%1000==0:
      valid_loss, valid_err = nn.get_loss(X, Y)
      print "Loss:",loss,"Valid Loss:",valid_loss,"Valid Error:",valid_err