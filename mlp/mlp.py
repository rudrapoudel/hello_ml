"""
An example of Multilayer Neural Network using stochastic
gradient descent.
"""
__docformat__ = 'restructedtext en'

import sys, os, random, cPickle, gzip

import numpy as np
rng = np.random.RandomState()


def load_mnist_data(filename = 'mnist.pkl.gz'):
  
  if (not os.path.isfile(filename)) and filename == 'mnist.pkl.gz':
    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, filename)
        
  f = gzip.open(filename, 'rb')
  train_data, validation_data, test_data = cPickle.load(f)
  f.close()
  
  return (train_data, validation_data, test_data)
 
class Activation():
  """
  Base class for the activation function.
  
  """
  def fn(self, z):
    """
    Return the activation 'a' for the input 'z'.
    
    """
    raise NotImplementedError()
  
  def delta(self, z):
    """
    Return the derivative of the activation function.
    
    """
    raise NotImplementedError()    
 
class SigmoidActivation(Activation):
  """
  The sigmoid function.
  
  """
  def fn(self, z):
    """
    Return the activation 'a' for the input 'z'.
    
    """
    return 1.0/(1.0+np.exp(-z))
  
  def delta(self, z):
    """
    Return the derivative of the activation function.
    
    """
    sz = self.fn(z)
    return sz * (1-sz)
   
class Cost():
  """
  Base class for the cost function.
  
  """
  
  def fn(self, a, y):
    """
    Return the cost associated with an network output (activation) 'a' and target 'y'.
    
    """
    raise NotImplementedError()
  
  def delta(self, z, a, y):
    """
    Return the derivative of the cost function for 'z', (activation) 'a' and target 'y'.
    
    """
    raise NotImplementedError()

class QuadraticCost(Cost):

  def __init__(self, activation = SigmoidActivation()):
    self.activation = activation
               
  def fn(self, a, y):
    """
    Return the cost associated with an network output (activation) 'a' and target 'y'.
    
    """
    return 0.5 * np.linalg.norm(a - y)**2 # 1/2 * ||a - y||^2

  def delta(self, z, a, y):
    """
    Return the derivative of the cost function for 'z', (activation) 'a' and target 'y'.
    
    """
    # Note, if your cost function is rather
    # 1/2 * ||y - a||^2, then deta will be,
    # -(y - a) * delta(z)
    # i.e. just exchange of the sign 
    return (a - y) * self.activation.delta(z)


class CrossEntropyCost(Cost):

  def fn(self, a, y):
    """
    Return the cost associated with an network output (activation) 'a' and target 'y'.
    Note: numpy.np.nan_to_num convert nan to 0, which might happen when a = y = 1
    
    """
    return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))

  def delta(self, z, a, y):
    """
    Return the derivative of the cost function for 'z', (activation) 'a' and target 'y'.
    
    """
    
    return (a - y) # For cross entropy activation delta/derivative term cancel out.
      
      
class MLP:
  
  def __init__(self, layers, W = None, b = None,
           w_init = None,
           b_init = None,
           activation = SigmoidActivation(),
           cost = CrossEntropyCost(),
           num_classes = 10):
    """
    Init MLP.
    layers- the number of nodes in each layers, example [784, 200, 10].
            First layer is also known as input layer and
            last layer is also known as output layer.
    W- weights matrix.
    b- biases.
    w_init- W init range
    b_init- b init value
    """
    self.num_layers = len(layers)
    self.layers = layers
    self.activation = activation
    self.cost = cost
    self.num_classes = num_classes
     
    # Init weights W if its null
    # The shape of the W is (n_out, n_in) i.e. w_ij means i<-j or
    # connection from node j in layer l-1 to node i in layer l.
    if not W:
      self.W = []
      for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
        # init weight range
        if w_init is None:
          w_bound = np.sqrt(6. / (n_in + n_out)) # good for tanh activation fn
          w_bound *= 4 # good for sigmoid activation fn
        else:
          w_bound = w_init
        w_values = np.asarray(rng.uniform(
                low = -w_bound,
                high = w_bound,
                size = (n_out, n_in)), dtype=np.float32)
        self.W.append(w_values)
    else:
      self.W = W
    #print self.W
       
    # Init biases b if its null
    if not b:
      if not b_init:
        self.b = [rng.randn(n_out, 1).astype(np.float32) for n_out in self.layers[1:]]
      else:
        self.b = [np.ones((n_out, 1), dtype = np.float32) * b_init for n_out in self.layers[1:]]
    else:
      self.b = b
    #print self.b

  def serialize(self, stream):
    """
    Save the MLP model i.e. weights and baises 
    """    
    cPickle.dump(self.W, stream, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(self.b, stream, protocol=cPickle.HIGHEST_PROTOCOL)

  def deserialize(self, stream):
    self.W = cPickle.load(stream)
    self.b = cPickle.load(stream)
          
  def feedforward(self, a):
    """ 
    Return the output of the network.
    """
    # TODO Optimization - this should be done in training set level
    a = a.reshape((len(a), 1))
    for w, b in zip(self.W, self.b):
        a = self.activation.fn(np.dot(w, a) + b)
    return a
  
  def vectorized_target(self, y):
    """
    Vectorized the target for argmax or cost functions
    """
    y_ = np.zeros((self.num_classes, 1))
    y_[y] = 1.0
    return y_
      
  def get_error(self, X, Y, argmax_y = False):
    """
    Report total error for the data.
    """
    m = X.shape[0]
    num_error = 0.0
    if argmax_y:
      for i in range(m):
        if np.argmax(self.feedforward(X[i])) != np.argmax(Y[i]):
          num_error +=1
    else:
      for i in range(m):
        if np.argmax(self.feedforward(X[i])) != Y[i]:
          num_error +=1
    return num_error / m
  
  def get_cost(self, X, Y, convert=False):
    """
    Report total cost for the data.
    """  
    vectorize_y = True
    if Y.ndim == 2:
      vectorize_y = False
      print 'vectorize_y = False'
      
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
      a = self.feedforward(X[i])
      if vectorize_y:
        y = self.vectorized_target(Y[i])
      cost += self.cost.fn(a, y)
      cost /=m

      # It's better to add L2 regularization cost to the final cost.
      # As L2 regularization has same effect as the weight decay,
      # i.e. weight_decay = l2_importance_factor / num_examples
    return cost
      
  def back_propagation(self, x, y):
    """
    Return a tuple "(grad_b, grad_w)" representing the
    gradient for the cost function. 
    """
    grad_w = [np.zeros(w.shape) for w in self.W]
    grad_b = [np.zeros(b.shape) for b in self.b]
    
    # TODO Optimization - this should be done in training set level
    x_ = x.reshape((len(x), 1))
    y_ = self.vectorized_target(y)
    
    # feedforward
    a = x_
    activations = [x_] # store all the activations
    zs = [] # store all the z vectors
    for w, b in zip(self.W, self.b):            
      z = np.dot(w, a) + b
      zs.append(z)
      a = self.activation.fn(z)
      activations.append(a)
    # backward pass    
    delta = self.cost.delta(zs[-1], activations[-1], y_)
    grad_w[-1] = np.dot(delta, activations[-2].transpose())
    grad_b[-1] = delta    
    
    for l in xrange(2, self.num_layers):
      z = zs[-l]
      z_delta = self.activation.delta(z)
      delta = np.dot(self.W[-l+1].transpose(), delta) * z_delta
      grad_b[-l] = delta
      grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (grad_w, grad_b)   
    
  def update_mini_batch(self, X, Y, learning_rate, weight_decay):
    """
    Training MLP using mini-batch stochastic gradient descent.
    """
    grad_w = [np.zeros(w.shape) for w in self.W]
    grad_b = [np.zeros(b.shape) for b in self.b]
    batch_size = len(Y)
    for i in range(batch_size):
      grad_w_i, grad_b_i = self.back_propagation(X[i], Y[i])
      grad_w = [gw+gwi for gw, gwi in zip(grad_w, grad_w_i)]
      grad_b = [gb+gbi for gb, gbi in zip(grad_b, grad_b_i)]
    self.W = [w - (weight_decay * learning_rate * w) - ((learning_rate / batch_size) * gw)
                  for w, gw in zip(self.W, grad_w)]
    self.b = [b - (learning_rate / batch_size) * gb 
                 for b, gb in zip(self.b, grad_b)]

  def train(self, train_data, validation_data,
            learning_rate,
            weight_decay = 0.0005,
            epochs = 10,
            minibatch_size = 32,
            verbose = True):
    """
    Training MLP using mini-batch stochastic gradient descent.
    """
    train_x = train_data[0]
    train_y = train_data[1]
    validation_x = validation_data[0]
    validation_y = validation_data[1]
    m = train_x.shape[0]
    n = train_x.shape[1]
    print 'Train dim: ', n, ' num examples: ', m
    print 'Valid dim: ', validation_x.shape[1], ' num examples: ', validation_x.shape[0]
  
    train_costs, train_errors = [], []
    validation_costs, validation_errors = [], []
    for iter in range(epochs):
      print "Epoch: ", iter+1
      new_inds = np.random.permutation(m)
      for minibatch_index in range(0, m, minibatch_size):
        minibatch_index_end = min(minibatch_index + minibatch_size, m)
        #print minibatch_index, ', ', minibatch_index_end
     
        train_batch_x = train_x[new_inds[minibatch_index:minibatch_index_end]]
        train_batch_y = train_y[new_inds[minibatch_index:minibatch_index_end]] 
        self.update_mini_batch(train_batch_x, train_batch_y, learning_rate, weight_decay)
        
      train_cost = self.get_cost(train_x, train_y)
      train_error = self.get_error(train_x, train_y)
      train_costs.append(train_cost)
      train_errors.append(train_error)
      
      validation_cost = self.get_cost(validation_x, validation_y)
      validation_error = self.get_error(validation_x, validation_y)
      validation_costs.append(validation_cost)
      validation_errors.append(validation_error)
      
      if verbose:
        print 'Training error  : {} , cost: {}'.format(train_error, train_cost)
        print 'Validation error: {} , cost: {}'.format(validation_error, validation_cost)
        
    return train_costs, train_errors, validation_costs, validation_errors

def run_example(load_saved_model = False,
                serialization_file = 'mlp.pkl',
                layers = [784, 500, 10], # size of input = 28 x 28 and ouput = 10.
                weight_decay = 0.0005,
                learning_rate = 0.01,
                epochs = 1000,
                minibatch_size = 128):  
  # Load dataset
  train_data, validation_data, test_data = load_mnist_data()
#   X = train_data[0]
#   Y = train_data[1]
#   
#   print X.shape
#   return;

  # Build MLP
  mlp = MLP(layers = layers,
            W = None,
            b = None,
            w_init = None,
            b_init = None) # defalut 0.1 is good start
  
  # Train mlp
  if load_saved_model:
    # deserialize W and b first
    print 'Loading MLP model ...'
    f = file(serialization_file, 'rb')
    mlp.deserialize(f)
    f.close()
    
    train_x = train_data[0]
    train_y = train_data[1]
    validation_x = validation_data[0]
    validation_y = validation_data[1]
    test_x = test_data[0]
    test_y = test_data[1]
    
    train_cost = mlp.get_cost(train_x, train_y)
    train_error = mlp.get_error(train_x, train_y)
    
    validation_cost = mlp.get_cost(validation_x, validation_y)
    validation_error = mlp.get_error(validation_x, validation_y)
    
    test_cost = mlp.get_cost(test_x, test_y)
    test_error = mlp.get_error(test_x, test_y)
    
    print 'Training error  : {} , cost: {}'.format(train_error, train_cost)
    print 'Validation error: {} , cost: {}'.format(validation_error, validation_cost)
    print 'Test error      : {} , cost: {}'.format(test_error, test_cost)
  else:
    print 'Training MLP model ...'
    train_costs, train_errors, validation_costs, validation_errors = mlp.train(
              train_data = train_data, validation_data = validation_data,
              weight_decay = weight_decay,
              learning_rate = learning_rate,
              epochs = epochs,
              minibatch_size = minibatch_size)
    # Save best W and b
    f = file(serialization_file, 'wb')
    mlp.serialize(f)
    f.close()
    print 'MLP model has been saved to ', serialization_file

def test():
  j = np.linalg.norm(0.5-1)
  j2 = j**2
  
  print j
  print j2

if __name__ == '__main__':
  
#   test()
  run_example(load_saved_model = False,
              serialization_file = 'mlp.pkl',
              layers = [784, 500, 10], # size of input = 28 x 28 and ouput = 10.
              weight_decay = 0.0005,
              learning_rate = 0.01,
              epochs = 100,
              minibatch_size = 128)
  
