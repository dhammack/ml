#show examples of usage for each classifier and regressor in this repo
#first generate some data
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelBinarizer
from numpy import *

from LogisticClassifier import *
from NeuralNetClassifier import *
from DeepNeuralNetClassifier import *
from LinearBasisRegressor import *


def error_rate(Yh, Y):
    return sum( not_equal(argmax(Yh,axis=1), argmax(Y,axis=1))) / float(Yh.shape[0]) 
	
def r_squared(Yh, Y):
	#1-(residual_sum_squares)/(mean_sum_squares)
	Ymean = mean(Y,axis=1)
	resid = sum((Y-Yh)**2)
	var = sum((Y-Ymean)**2)
	return 1-resid/var
	
def fourier_basis(x):
    #use sine waves with different amplitudes
    sins =  hstack(tuple(sin(pi*n*x)) for n in arange(0,1,0.1))
    coss = hstack(tuple(cos(pi*n*x)) for n in arange(0,1,0.1))
    return hstack((sins, coss))
	
if __name__ == '__main__':
	#Classification
	X,Y = make_classification(n_features=20, n_informative=8, n_redundant=12,
							n_repeated=0, n_classes=5, n_clusters_per_class=2)
	#make Y into a one-hot matrix, the format we use.
	lb = LabelBinarizer()
	Y = lb.fit_transform(Y)
	
	#initialize a Logistic Regression Classifier
	log_reg = LogisticClassifier(basis='rectifier')
	#train
	log_reg.fit(X, Y, itrs=500, learn_rate=0.1, reg=1e-4, momentum=0.9, 
				proj_layer_size=50)
	
	print 'Logistic Classifier error rate:', error_rate(log_reg.predict(X), Y)
	#of course, use a separate train/test set in practice.
	
	#initialize a regular neural net classifier
	nnet = NeuralNetClassifier(layer_one_size=50)
	nnet.fit(X, Y, itrs=500, learn_rate=0.1, reg=1e-4, momentum=0.9)
	print 'Neural Network (shallow) error rate:', error_rate(nnet.predict(X), Y)
	
	#deeper neural network
	deepnet = DeepNeuralNetClassifier(layer_sizes=[100,100], dropout_rate=0.5)
	deepnet.fit(X, Y, itrs=2000, learn_rate=0.1, reg=1e-7, momentum=0.9,
				batch_size=10)
	print 'Deep (100-100) Neural Network error rate', error_rate(deepnet.predict(X), Y)
	
	#regression
	X = uniform(-5, 5, size=(200,2))
	Y = sin(X[:,0]) + X[:,1]
	Y = Y.reshape((-1,1)) #make into column vector
	
	lin_reg = LinearBasisRegressor()
	lin_reg.fit_gd(X, Y, itrs=500, learning_rate=0.1, regularization=1e-4)
	print 'R**2 of the LinearBasisRegressor, using a linear basis', r_squared(lin_reg.predict(X), Y)
	
	#showing how to use a custom basis
	fourier_reg = LinearBasisRegressor(basisfunc=fourier_basis)
	fourier_reg.fit_gd(X, Y, itrs=500, learning_rate=0.1, regularization=1e-2)
	print 'R**2 of the LinearBasisRegressor, using a fourier basis', r_squared(fourier_reg.predict(X), Y)
	
	raw_input('press ENTER to exit')
	
	