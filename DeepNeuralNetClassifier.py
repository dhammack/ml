from numpy import *
from numpy.random import *

class DeepNeuralNetClassifier(object):
	"""
	Deep (at least one hidden layer) neural network classifier.
	You can specify custom basis functions for the hidden layers, 
	output is a softmax classifier. 
	"""
		
	def __init__(self, layer_sizes, activation=None, dropout_rate=0.5, dropout_input=0.2):
		"""
		Instantiate a single-hidden layer neural network
		layer_sizes - number and sizes of hidden layers.
		dropout_rate - expected percentage of units to drop during training.
		"""
		self.W = [] #array of weights for each layer.
		self.layer_sizes = layer_sizes
		self.dropout_rate = dropout_rate
		self.dropout_input = dropout_input
		if activation == None:
			activation = self.rectifier
		
		self.activation = activation
		
	def add_bias(self, X):
		#add a dummy feature of ones
		return hstack((X,ones((X.shape[0],1))))
	
	def fit(self, X, Y, itrs=100, learn_rate=0.1, reg=0.1,
			momentum=0.5, report_cost=False, batch_size=-1):
		"""
		Fit the model. 
		X - observation matrix (observations by dimensions)
		Y - one-hot target matrix (examples by classes)
		itrs - number of iterations to run
		learn_rate - size of step to use for gradient descent
		reg - regularization penalty
		momentum - weight of the previous gradient in the update step
		report_cost - if true, return the loss function at each step (expensive).
		batch_size - size of minibatches to use in training
		"""
		
		if batch_size==-1:
			batch_size=X.shape[0]
		
		#add a bias term (so the mean can be nonzero)
		X = self.add_bias(X)
		
		#hidden layers
		self.W_hid = []
		for insize, outsize in zip([X.shape[1]] + self.layer_sizes, self.layer_sizes):
			self.W_hid.append( uniform(-0.01, 0.01, (outsize, insize)) )
			
		#output layer (softmax classifier)
		self.W_out = uniform(-0.3, 0.3, (Y.shape[1], self.layer_sizes[-1]))
		
		#optimize
		costs = []
		layer_grads_prev = [zeros(W.shape) for W in self.W_hid] +  [zeros(self.W_out.shape)]
		for i in range(itrs):
			#get batch
			minibatch_inds = self.batch_inds(batch_size, X.shape[0])
			
			#compute gradients (uses backprop)
			layer_grads = self.grad(X[minibatch_inds,:], Y[minibatch_inds,:], reg) 
			
			#update hidden layers
			self.W_hid = [W_i - learn_rate*(grad_i + momentum*prev_grad_i) 
					  for W_i, grad_i, prev_grad_i in 
					  zip(self.W_hid, layer_grads[:-1], layer_grads_prev[:-1])]
			#update output layer
			self.W_out = self.W_out - learn_rate*(layer_grads[-1] + momentum*layer_grads_prev[-1])
			
			#update the momentum terms
			layer_grads_prev = layer_grads
			
			if report_cost:
				costs.append(self.loss(X,Y,reg))
 
		return costs
	
	def batch_inds(self, batch_size, data_size):
		inds = permutation(data_size)[:batch_size]
		return inds
	
	def rectifier(self, X, W):
		#returns the activations and grad (wrt W)
		Z = dot(X,W.T)
		act = maximum(0,Z)
		grad = greater(act,0)
		return act, grad

	def softmax(self, X, W):
		#softmax activation function
		Z = dot(X, W.T)
		Z = maximum(Z, -1e3) #capped for numerical reasons
		Z = minimum(Z, 1e3)
		numerator = exp(Z)
		S = numerator / sum(numerator, axis=1).reshape((-1,1))
		grad = S*(1-S)
		return S, grad    

	
	#used in development.
	
#     def grad_check(self, X, Y, reg):
#         inds = [(0,0), (2,2), (1,2), (0,2)]
#         layer = 0
#         X = self.add_bias(X)
#         layer_grads = self.grad(X, Y, reg)
#         for ind in inds:
#             grad_calc = layer_grads[layer][ind]
#             grad_numer = self.numeric_grad(X, Y, reg, layer, ind)
#             print 'calculated grad:', grad_calc, 'numeric grad:', grad_numer
#             print 'ratio:', grad_calc / grad_numer, 'diff:', grad_calc - grad_numer
			
#     def numeric_grad(self, X, Y, reg, layer, index=(0,0)):
#         #compute the numeric gradient for a given layer and index.
#         W_copy = copy(self.W_hid[layer])
#         #central difference method
#         ep = 1e-5
#         self.W_hid[layer][index] += ep
#         left_loss = self.loss(X, Y, reg)
#         self.W_hid[layer][index] -= 2*ep
#         right_loss = self.loss(X, Y, reg)
		
#         grad = (left_loss-right_loss)/2/ep
#         self.W_hid[layer] = W_copy
#         return grad
	
	
	def loss(self, X, Y, reg):
		#loss function to minimize.
		Yh = self.predict(X, add_bias=False)
		#regularization
		hidden_total = 0.5*reg*sum(sum(sum(W_hid_i**2)) for W_hid_i in self.W_hid)
		output_total = 0.5*reg*sum(sum(self.W_out**2))
		return mean(mean(-Y*log(Yh))) + hidden_total + output_total
	
	

		
	def predict(self, X, add_bias=True):
		"""
		If the model has been trained, makes predictions on an observation matrix (observations by features)
		"""
		#feed forward
		if add_bias:
			X = self.add_bias(X)
		
		#adjust for dropout
		X = X*(1-self.dropout_input)
		
		for W in self.W_hid:
			X, dX = self.activation(X*(1-self.dropout_rate), W)
			
		#make a prediction on top
		Y, dY = self.softmax(X*(1-self.dropout_rate), self.W_out)
		return Y

	def grad(self, X, Y, reg):
		"""
		Returns an array. First element is the gradient wrt the layer 1 weights, and the
		second element is the gradient wrt the layer 2 weights. 
		"""
		grads = []
		#feed forward step. Save the transformed X values because we'll need them in backprop
		Xs = [X]
		dXs = [1]
		dropout_masks = []
		for W in self.W_hid:
			d_rate = self.dropout_rate if len(Xs) > 1 else self.dropout_input
			dropout_masks.append(binomial(n=1, p=(1-d_rate), size=Xs[-1].shape))
			Xn, dXn = self.activation(Xs[-1]*dropout_masks[-1], W)
			Xs.append(Xn)
			dXs.append(dXn)
		
		#generate a dropout mask for the outputs as well
		dropout_masks.append(binomial(n=1, p=(1-self.dropout_rate), size=Xs[-1].shape))
		Y_hat, dY = self.softmax(Xs[-1]*dropout_masks[-1], self.W_out)
		
		#now compute gradients (back prop)
		delta = Y-Y_hat
		for Wi, Xi, dXi, masks_i in reversed(zip(self.W_hid + [self.W_out],Xs, dXs, dropout_masks)):
			grads.append( -dot(delta.T, Xi*masks_i)/X.shape[0]/Y.shape[1] + reg*Wi) #average over training set
			delta = dot(delta, Wi)*dXi #updating the deltas (chain rule)
		
		return list(reversed(grads))

	def fit_with_valid(self, X, Y, Xv, Yv, itrs=100, learn_rate=0.1, reg=0.1,
			momentum=0.9, batch_size=-1):
		"""
		Fit the model. 
		X - observation matrix (observations by dimensions)
		Y - one-hot target matrix (examples by classes)
		Xv - validation observations
		Yv - validation labels
		itrs - number of iterations to run
		learn_rate - size of step to use for gradient descent
		reg - regularization penalty (lambda above)
		momentum - weight of the previous gradient in the update step
		report_cost - if true, return the loss function at each step (expensive).
		batch_size - size of minibatches to use in training
		"""
		
		if batch_size==-1:
			batch_size=X.shape[0]
		
		#add a bias term (so the mean can be nonzero)
		X = self.add_bias(X)
		Xv = self.add_bias(Xv)
		
		#hidden layers
		self.W_hid = []
		for insize, outsize in zip([X.shape[1]] + self.layer_sizes, self.layer_sizes):
			self.W_hid.append( uniform(-0.01, 0.01, (outsize, insize)) )
			
		#output layer (softmax classifier)
		self.W_out = uniform(-0.3, 0.3, (Y.shape[1], self.layer_sizes[-1]))
		
		#optimize
		train_costs = []
		valid_costs = []
		layer_grads_prev = [zeros(W.shape) for W in self.W_hid] +  [zeros(self.W_out.shape)]
		for i in range(itrs):
			#get batch
			minibatch_inds = self.batch_inds(batch_size, X.shape[0])
			
			#compute gradients (uses backprop)
			layer_grads = self.grad(X[minibatch_inds,:], Y[minibatch_inds,:], reg) 
			
			#update hidden layers
			self.W_hid = [W_i - learn_rate*(grad_i + momentum*prev_grad_i) 
					  for W_i, grad_i, prev_grad_i in 
					  zip(self.W_hid, layer_grads[:-1], layer_grads_prev[:-1])]
			#update output layer
			self.W_out = self.W_out - learn_rate*(layer_grads[-1] + momentum*layer_grads_prev[-1])
			
			#update the momentum terms
			layer_grads_prev = layer_grads
			if i % 5 == 0:
				train_costs.append(self.loss(X,Y,reg))
				valid_costs.append(self.loss(Xv, Yv, reg))
			
		return train_costs, valid_costs
