'''
#------------------------------------------------------------------------------#
#--------------------------------- dpln::nnhd ---------------------------------#
#------------------------- author: gyang274@gmail.com -------------------------#
#------------------------------------------------------------------------------#
'''
#------------------------------------------------------------------------------#
#----- netural network and deep learning - handwritten digit recongnition -----#
#------------------------------------------------------------------------------#

#--------+---------+---------+---------+---------+---------+---------+---------#
#234567890123456789012345678901234567890123456789012345678901234567890123456789#

#------------------------------------------------------------------------------#
#------------------------------------ init ------------------------------------#
#------------------------------------------------------------------------------#
import random
import numpy as np
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#----------------------------------- class ------------------------------------#
#------------------------------------------------------------------------------#

#------------------------------- class Network --------------------------------#

# sizes is a list contains the number of neurons in the respective layers: e.g.,
# to create a Network object with 2 neurons in the first layer, 3 neurons in the 
# second layer, and 1 neuron in the final layer: net = Network([2, 3, 1]).
class Network(object):

  def __init__(self, sizes):
    
    self.num_layers = len(sizes)
    
    self.sizes = sizes
    
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#

# weights and biases are stored as lists of Numpy matrices, e.g., net.weights[1] 
# stores the weights connecting the second and third layers of neurons.

# denote w_1 = net.weights[1], then w_1_jk is the weight for the connection btwn
# the k_th neuron in the 2nd layer and the j_th neuron in the third layer.

# -> a' = \sigma(w \cdot a + b)

# a' - the vector of activations of the third layer of neurons can be obtained by 
# multiplying a - the vector of activations of the second layer of neurons, by w 
# - the weight matrix, and add the vector b - biases, and then apply the function 
# \sigma elementwise to every entry in the vector w a + b.

#------------------------------------------------------------------------------#

# sigmoid: sigmoid function.
def sigmoid(z): 
  return 1.0 / (1.0 + np.exp(-z))
#

# when the input z is a vector or a Numpy array, Numpy automatically applies the 
# function sigmoid elementwise, that is, in vectorized form.

#------------------------------------------------------------------------------#

# feedforward: apply sigmoid on each layer
def feedforward(self, a):
  '''Return the output of the network if "a" is input.'''
  for w, b in zip(self.weights, self.biases):
    a = sigmoid(np.dot(w, a)+b)
  return a
#

# this assumes that the input a is an (n, 1) Numpy ndarray - not an (n, ) vector 
# here, n is the number of inputs to the network. although using an (n, ) vector 
# appears the more natural choice, using an (n, 1) ndarray makes it particularly
# easy and convenient to modify the codes to feedforward multiple inputs at once

#---------------------- SGD: stochastic gradient descent ----------------------#

def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
  
  '''Train the neural network using mini-batch stochastic gradient descent.
  The "training_data" is a list of tuples "(x, y)" representing the training
  inputs and the desired outputs. The other non-optional parameters are self
  explanatory. If "test_data" is provided then the network will be evaluated
  against the test data after each epoch, and partial progress printed out.
  This is useful for tracking progress, but slows things down substantially.'''
  
  if test_data: n_test = len(test_data)
  
  n = len(training_data)
  
  for j in xrange(epochs):
    
    random.shuffle(training_data)
    
    mini_batches = [
      training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)
    ]
    
    for mini_batch in mini_batches:
      
      self.update_mini_batch(mini_batch, eta)
      
    if test_data:
      
      print 'Epoch {0}: {1} / {2}'.format(j, self.evalute(test_data), n_test)
      
    else:
      
      print 'Epoch {0} complete..'.format(j)
  
  return None
#

def update_mini_batch(self, mini_batch, eta):
  
  '''Update the network's weights and biases by applying gradient descent using
  backpropagation to a single mini_batch. The "mini_batch" is a list of tuples
  "(x, y)", and "eta" is the learning rate.'''

  nabla_w = [np.zeros(w.shape) for w in self.weights]
  
  nabla_b = [np.zeros(b.shape) for b in self.biases]
  
  for x, y in mini_batch:
    
    '''call backpropagation algorithm to compute the gradient of the cost function.'''
    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
    
    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    
    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
  self.weights = [
    w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
  ]
  
  self.biases = [
    b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
  ]
  
  return None
#

#------------------------------------------------------------------------------#
