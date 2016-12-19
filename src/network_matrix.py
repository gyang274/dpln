'''
#------------------------------------------------------------------------------#
#--------------------------------- dpln::bkpg ---------------------------------#
#------------------------- author: gyang274@gmail.com -------------------------#
#------------------------------------------------------------------------------#
'''
#------------------------------------------------------------------------------#
#-------- netural network and deep learning - backpropagation algorithm -------#
#------------------------------------------------------------------------------#
'''
A module to implement the stochastic gradient descent learning algorithm for a
feedforward neural network.  Gradients are calculated using backpropagation.
Note that I have focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

+ modify network.py with fully matrix-based approach to backpropagation over a
mini-batch - improve the speed on mnist classification problem by a factor of 2
'''
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
class Network(object):
  
  def __init__(self, sizes):
    """The list `sizes` contains the number of neurons in the respective layers
    of the network.  For example, if the list was [2, 3, 1] then it would be a
    three-layer network, with the first layer containing 2 neurons, the second
    layer 3 neurons, and the third layer 1 neuron.  The biases and weights for
    the network are initialized randomly, using a Gaussian distribution with a
    mean 0, and variance 1. Note that the first layer is assumed to be an input
    layer, and by convention we won't set any biases for those neurons, since
    biases are only ever used in computing the outputs from the later layers."""
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.weights = [
      np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
    ]
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
  #
  
  def feedforward(self, a):
    """Return the output of the network if `a` is input."""
    for w, b in zip(self.weights, self.biases):
      a = sigmoid(np.dot(w, a)+b)
    return a
  #

  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Train the neural network using mini-batch stochastic gradient descent.
    The `training_data` is a list of tuples `(x, y)` representing the training
    inputs and the desired outputs. The other non-optional parameters are self
    explanatory. If `test_data` is provided then the network will be evaluated
    against the test data after each epoch, and partial progress printed out.
    This is useful for tracking progress, but slows things down substantially."""
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
        print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
      else:
        print "Epoch {0} complete".format(j)
  #

  def update_mini_batch(self, mini_batch, eta):
    """Update the network's weights and biases by applying gradient descent alg
    using backpropagation to a single mini batch. The `mini_batch` is a list of
    tuples `(x, y)`, and `eta` is the learning rate."""
    # sequeeze x and y in mini_batch into matrix
    x = np.array([m[0] for m in mini_batch])
    y = np.array([m[1] for m in mini_batch])
    # calculate nabla_w, nabla_b using backpropagation
    nabla_w, nabla_b = self.backprop_mini_batch(x, y)
    # update weights and biases using nabla_w, nabla_b from mini_batch
    self.weights = [
      w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)
    ]
    self.biases = [
      b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)
    ]
  #

  def backprop(self, x, y):
    """Return a tuple `(nabla_w, nabla_b)` representing the gradient for the
    cost function C_x. `nabla_w` and `nabla_b` are layer-by-layer lists of numpy
    arrays, similar to `self.weights` and `self.biases`."""
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little differently to
    # the notation in Chapter 2 of the book. Here, l = 1 means the last layer of
    # neurons, and l = 2 is the second-last layer, and so on. It's a renumbering 
    # of the scheme in the book, used here to take advantage of the fact that is
    # Python can use negative indices in lists.
    for l in xrange(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_w, nabla_b)
  #

  def backprop_mini_batch(self, x, y):
    """Return a tuple `(nabla_w, nabla_b)` representing the averge of gradient
    for the cost function C_{x_1}, ..., C_{x_m}. `nabla_w` and `nabla_b` are
    layer-by-layer lists of numpy arrays, similar to `self.weights` and
    `self.biases`."""    
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation).swapaxes(0, 1)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)    
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = np.sum(delta, axis=0)
    nabla_w[-1] = np.sum(np.array([np.dot(d, a.transpose()) for d, a in zip(delta, activations[-2])]), axis=0)
    # Note that the variable l in the loop below is used a little differently to
    # the notation in Chapter 2 of the book. Here, l = 1 means the last layer of
    # neurons, and l = 2 is the second-last layer, and so on. It's a renumbering 
    # of the scheme in the book, used here to take advantage of the fact that is
    # Python can use negative indices in lists.
    for l in xrange(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta).swapaxes(0, 1) * sp
      nabla_b[-l] = np.sum(delta, axis=0)
      nabla_w[-l] = np.sum(np.array([np.dot(d, a.transpose()) for d, a in zip(delta, activations[-l-1])]), axis=0)
    return (nabla_w, nabla_b)
  #
    
  def evaluate(self, test_data):
    """Return the number of test inputs for which the neural network outputs the
    correct result. Note that the neural network's output is assumed to be the
    index of whichever neuron in the final layer has the highest activation."""
    test_results = [
      (np.argmax(self.feedforward(x)), y) for (x, y) in test_data
    ]
    return sum(int(x == y) for (x, y) in test_results)
  #
  
  def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x / \partial a for
    the output activations."""
    return (output_activations-y)
  #
  
#-------------------------- miscellaneous functions ---------------------------#
def sigmoid(z):
  """The sigmoid function."""
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z)*(1-sigmoid(z))
#------------------------------------------------------------------------------#

'''
#------------------------------------------------------------------------------#
#------------------------------------ main ------------------------------------#
#------------------------------------------------------------------------------#
'''
#------------------------------------ main ------------------------------------#
if __name__ == "__main__":
  
  import mnist_loader
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  
  import network_matrix
  net = network_matrix.Network([784, 30, 10])
  net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#
#------------------------------------------------------------------------------#
