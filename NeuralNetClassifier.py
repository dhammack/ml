from numpy import *
from numpy.random import *


class NeuralNetClassifier(object):
	"""
	Neural network classifier. Has one hidden (projection) layer with 
	adaptable weights. Default activation function is the rectifier.
	After the rectifier layer, a logistic classifier is learned.
	"""
	
	def __init__(self, layer_one_size=10, activation=None):
		"""
		Instantiate a single-hidden layer neural network
		activation - function taking in a bunch of row vectors (a matrix) 
			and a weight matrix. It must compute the activation and the gradient
			and return both.
		"""
		self.W_hid = array([]) #basis mapping layer
		self.W_out = array([]) #logistic classifier layer
		self.layer_one_size = layer_one_size
		if activation == None:
			activation = self.rectifier
		
		self.activation = activation
		
	def add_bias(self, X):
		#adds a dummy column of ones to a matrix. Allows for fitting noncentered data.
		return hstack((X,ones((X.shape[0],1))))
	
	def fit(self, X, Y, itrs=100, learn_rate=0.1, reg=0.1,
			momentum=0.5, report_cost=False, batch_size=-1):
		"""
		Fit the model. Returns nothing unless report_cost == True.
		If cost reporting is turned on, return the costs and L2 norm of the gradient over time.
		X - observation matrix (examples by features)
		Y - one-hot target matrix (examples by classes)
		itrs - number of iterations to run
		learn_rate - size of step to use for gradient descent
		reg - regularization penalty
		momentum - fraction of the previous gradient used in the update step
		report_cost - if true, return the loss function at each step (expensive).
		batch_size - size of minibatches to use in training. -1 means full batch.
		"""
		
		if batch_size==-1:
			batch_size=X.shape[0]
		
		Xb = self.add_bias(X)
		#these should be initialized based on the square root of the fan in, but this should be OK.
		self.W_hid = uniform(-0.1, 0.1, (self.layer_one_size, Xb.shape[1]))
		self.W_out = uniform(-0.1, 0.1, (Y.shape[1], self.layer_one_size))
		#set up for learning
		costs = []
		grad_norms = []
		layer_grads = [zeros(self.W_hid.shape), zeros(self.W_out.shape)]
		layer_grads_prev = [zeros(self.W_hid.shape), zeros(self.W_out.shape)]
		#learn.
		for i in range(itrs):
			minibatch_inds = self.batch_inds(batch_size, X.shape[0]) #get a minibatch
			layer_grads = self.grad(Xb[minibatch_inds,:], Y[minibatch_inds,:], reg) #compute gradients (uses backprop)
			#update the weights
			self.W_hid = self.W_hid - learn_rate*(layer_grads[0] + momentum*layer_grads_prev[0])
			self.W_out = self.W_out - learn_rate*(layer_grads[1] + momentum*layer_grads_prev[1])
			#update the momentum term
			layer_grads_prev = layer_grads
			
			if report_cost:
				costs.append(self.loss(X,Y,reg))
				grad_norms.append(norm(layer_grads[0]) + norm(layer_grads[1]))
		
		return costs, grad_norms
	
	def batch_inds(self, batch_size, data_size):
		#given a batch size and the size of the data, get a minibatch
		inds = permutation(data_size)[:batch_size]
		return inds
	
	def softmax(self, X, W):
		#softmax activation function
		Z = dot(X, W.T)
		Z = maximum(Z, -1e3)
		Z = minimum(Z, 1e3)
		numerator = exp(Z)
		S = numerator / sum(numerator, axis=1).reshape((-1,1))
		grad = S*(1-S)
		return S, grad
	
	def predict(self, X, add_bias=True):
		"""
		If the model has been trained, makes predictions on an observation matrix (observations by features)
		"""
		if add_bias:
			X = self.add_bias(X)
			
		#map to our learned basis. Ignore the gradient.
		X2, dX2 = self.activation(X, self.W_hid)
		
		#make a prediction on top
		Y, dY = self.softmax(X2, self.W_out)
		return Y
	
	def rectifier(self, X, W):
		#rectifier activation function.
		#returns max(0, Wx), and the gradient
		Z = dot(X,W.T)
		act = maximum(0,Z)
		grad = greater(act,0)
		return act, grad
		
	def grad(self, X, Y, reg):
		"""
		Returns an array. First element is the gradient wrt the layer 1 weights, and the
		second element is the gradient wrt the layer 2 weights. 
		"""
		layers = [] #will hold the gradients of each layer of weights.
		
		#feed forward pass
		X_2, X_2_grad = self.activation(X, self.W_hid)
		Yh, dYh = self.softmax(X_2, self.W_out)
		
		#now compute gradients (back prop)
		delta = Y-Yh #Take advantage of the cancellation of terms in CE loss + softmax
		
		#gradient is averaged over all training examples and classes
		W_out_grad = -dot(delta.T, X_2)/X.shape[0]/Y.shape[1]
		layers.append( W_out_grad + reg*self.W_out) #include regularization
		
		#update our delta, using the chain rule.
		delta = dot(delta, self.W_out)*X_2_grad
		
		#again, average over all examples + classes. Add a regularization term.
		W_hid_grad = -dot(delta.T, X)/X.shape[0]/Y.shape[1]
		layers.append(W_hid_grad + reg*self.W_hid)
		
		#backprop is...backwards. Reverse it.
		return list(reversed(layers))
	
	def loss(self, X, Y, reg, add_bias=True):
		#Loss function. Used internally for reporting.
		Yh = self.predict(X, add_bias)
		reg_W_hid = 0.5*reg*sum(sum(self.W_hid**2))
		reg_W_out = 0.5*reg*sum(sum(self.W_out**2))
		return mean(mean(-Y*log(Yh))) + reg_W_out + reg_W_hid
	
	#these were used during development.
#     def grad_check(self, X, Y, reg):
#         inds = [(0,0), (2,2), (1,2), (0,2)]
#         layer = 1
#         X = self.add_bias(X)
#         layer_grads = self.grad(X, Y, reg)
#         for ind in inds:
#             grad_calc = layer_grads[layer][ind]
#             grad_numer = self.numeric_grad(X, Y, reg, ind)
#             print 'calculated grad:', grad_calc, 'numeric grad:', grad_numer
#             print 'ratio:', grad_calc / grad_numer, 'diff:', grad_calc - grad_numer
			
#     def numeric_grad(self, X, Y, reg, index=(0,0)):
#         #compute the numeric gradient for a given layer and index.
#         #W_copy = copy(self.W_hid)
#         W_copy = copy(self.W_out)
#         #central difference method
#         ep = 1e-5
#         #self.W_hid[index] += ep
#         self.W_out[index] += ep
#         left_loss = self.loss(X, Y, reg, add_bias=False)
#         #self.W_hid[index] -= 2*ep
#         self.W_out[index] -= 2*ep
#         right_loss = self.loss(X, Y, reg, add_bias=False)
		
#         grad = (left_loss-right_loss)/2/ep
#         #self.W_hid = W_copy
#         self.W_out = W_copy
#         return grad