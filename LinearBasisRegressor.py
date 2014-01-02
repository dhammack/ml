from numpy import *
from numpy.random import *

class LinearBasisRegressor(object):
	"""
	A generic linear regressor. Uses nonlinear basis functions, can fit with
	either the normal equations or gradient descent
	"""
	
	def __init__(self, basisfunc=None):
		"""
		Instantiate a linear regressor. If you want to use a custom basis function,
		specify it here. It should accept an array and output an array. The default
		basis function is the identity function.
		"""
		self.w = array([])
		self.basisfunc = basisfunc if basisfunc is not None else self.identity
		
	def identity(self, x):
		#identity basis function - for linear models in x
		return x
	
	def basismap(self, X):
		#return X in the new basis (the design matrix)
		Xn = zeros((X.shape[0], self.basisfunc(X[0,:]).shape[0]))
		for i, xi in enumerate(X):
			Xn[i,:] = self.basisfunc(xi)
		return Xn
	
	def fit_gd(self, X, y, itrs=100, learning_rate=0.1, regularization=0.1):
		"""
		fit using iterative gradient descent with least squares loss
		itrs - iterations of gd
		learning_rate - learning rate for updates
		regularization - weight decay. Greated values -> more regularization
		"""
		
		if len(X.shape) == 1: #in case a 1-d array was specified, make it a column vector
			X = X.reshape((-1,1))
		
		#first get a new basis by using our basis func
		Xn = self.basismap(X)

		#initial weights
		self.w = uniform(-0.1, 0.1, (Xn.shape[1],1))
		
		#now optimize in this new space, using gradient descent
		
		for i in range(itrs):
			grad = self.grad(Xn, y, regularization)
			self.w = self.w - learning_rate*grad
		
	def grad(self, X, y, reg):
		"""
		Returns the gradient of the loss function with respect to the weights.
		Used in gradient descent training.
		"""
		return  -mean((y - dot(X, self.w)) * X, axis=0).reshape(self.w.shape) + reg*self.w
	
	def fit_normal_eqns(self, X, y, reg=1e-5):
		"""
		Solves for the weights using the normal equation. 
		"""
		#if a 1d array was specified, make it a column vector
		if len(X.shape) == 1:
			X = X.reshape((-1,1))
			
		Xn = self.basismap(X)
		#self.w = dot(pinv(Xn), y)
		self.w = dot(dot(inv(eye(Xn.shape[1])*reg + dot(Xn.T, Xn)), Xn.T) , y)
	
	def predict(self, X):
		"""
		Makes predictions on a matrix of (observations x features)
		"""
		Xn = self.basismap(X)
		return dot(Xn, self.w)
	
	def loss(self, X, y):
		#assumes that X is the data matrix (not the design matrix)
		yh = self.predict(X)
		return mean((yh-y)**2)
	