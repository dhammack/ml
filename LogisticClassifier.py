from numpy import *
from numpy.random import *

class LogisticClassifier(object):
	"""
	Multiclass logistic regression with regularization. Trained with gradient descent + momentum (if desired). 
	"""
	
	def __init__(self, basis=None):
		"""
		Instantiate a logistic regression model. Options for the basis are
		'poly', 'rbf', 'sigmoid', 'rectifier', or 'linear'.
		"""
		self.W = array([])
		self.A = None #the mixing matrix for basis mapping.
		self.basis=basis
		if basis == 'poly':
			self.basisfunc = self.poly_basis
		elif basis == 'rbf':
			self.basisfunc = self.rbf_basis
		elif basis == 'sigmoid':
			self.basisfunc = self.sigmoid_basis
		elif basis == 'rectifier':
			self.basisfunc = self.rectifier_basis
		else:
			self.basisfunc = self.identity
			
	
	def identity(self, x):
		#identity basis function + a bias
		return hstack((x,1))
	
	def poly_basis(self, x):
		#polynomial basis
		degree = 2
		#first mix the components of x in a higher dimension
		xn = dot(self.A,x)
		return self.identity(hstack(tuple(sum(xn**i for i in range(degree)))))
		
	def rbf_basis(self, x):
		#in this case, use the mixing matrix as centroids.
		return self.identity(hstack(tuple(exp(-norm(x-mu)) for mu in self.A)))
	
	def sigmoid_basis(self, x):
		#just like a neural network layer.
		xn = dot(self.A, x)
		return self.identity((1+exp(-xn))**-1)
	
	def rectifier_basis(self, x):
		#used in the latest neural nets
		xn = dot(self.A, x)
		return self.identity(maximum(xn, 0))
	
	def basismap(self, X):
		#if X is an observation matrix (examples by dimensions),
		#return each row mapped to a higher dimsional space
		new_dimensions = self.basisfunc(X[0,:]).shape[0]
		Xn = zeros((X.shape[0], new_dimensions))
		for i,xi in enumerate(X):
			Xn[i,:] = self.basisfunc(xi)
		return Xn
	
	def fit(self, X, Y, itrs=100, learn_rate=0.1, reg=0.1,
			momentum=0.5, report_cost=False, proj_layer_size=10):
		"""
		Fit the model. 
		X - observation matrix (observations by dimensions)
		Y - one-hot target matrix (examples by classes)
		itrs - number of iterations to run
		learn_rate - size of step to use for gradient descent
		reg - regularization penalty (lambda above)
		momentum - weight of the previous gradient in the update step
		report_cost - if true, return the loss function at each step (expensive).
		proj_layer_size - number of dimensions in the projection (mixing) layer. Higher -> more variance
		"""
		
		#first map to a new basis
		if self.basis != 'rbf':
			self.A = uniform(-1, 1, (proj_layer_size, X.shape[1]))
		else:
			#use the training examples as bases
			self.A = X[permutation(X.shape[0])[:proj_layer_size],:]
		Xn = self.basismap(X)
		
		#set up weights
		self.W = uniform(-0.1, 0.1, (Y.shape[1], Xn.shape[1]))
 
		#optimize
		costs = []
		previous_grad = zeros(self.W.shape) #used in momentum
		for i in range(itrs):
			grad = self.grad(Xn, Y, reg) #compute gradient
			self.W = self.W - learn_rate*(grad + momentum*previous_grad) #take a step, use previous gradient as well.
			previous_grad = grad
			
			if report_cost:
				costs.append(self.loss(X,Y,reg))
		
		return costs
	
	def softmax(self, Z):
		#returns sigmoid elementwise
		Z = maximum(Z, -1e3)
		Z = minimum(Z, 1e3)
		numerator = exp(Z)
		return numerator / sum(numerator, axis=1).reshape((-1,1))
	
	def predict(self, X):
		"""
		If the model has been trained, makes predictions on an observation matrix (observations by features)
		"""
		Xn = self.basismap(X)
		return self.softmax(dot(Xn, self.W.T))
	
	def grad(self, Xn, Y, reg):
		"""
		Returns the gradient of the loss function wrt the weights. 
		"""
		#Xn should be the design matrix
		Yh = self.softmax(dot(Xn, self.W.T))
		return -dot(Y.T-Yh.T,Xn)/Xn.shape[0] + reg*self.W
	
	def loss(self, X, Y, reg):
		#assuming X is the data matrix
		Yh = self.predict(X)
		return -mean(mean(Y*log(Yh))) - reg*trace(dot(self.W,self.W.T))/self.W.shape[0]
	