from numpy import *
import numpy as np
import warnings
from matplotlib import pyplot as plt
import pdb

eps = 1e-40



def linearBasis(x):

	n_points_x = x.shape[0]
	X = concatenate( (ones((n_points_x,1)),x),axis=1 )
	return X

def polynomialCustomBasis(x):

	n_points_x = x.shape[0]
	X = concatenate( (ones((n_points_x,1)),x,x**3,x**5),axis=1 )
	return X




def fit(x,y,s="none",expand=linearBasis,empirical_errors=False,unit_box=False,invert_x=False,eps=1e-40):

	(x,y,s) = checkDimsData(x,y,s)

	if unit_box==True:
		x_nobox = x.copy()
		x, min_x, max_x_kron = to_unit_box(x,invert_x)

	X=expand(x)
	n_dims_X=X.shape[1]

	ys=y/s
	Xs=X/kron(ones((n_dims_X,1)),s).T

	XX = dot(Xs.T,Xs) + eps*identity(n_dims_X)
	Xy = dot(Xs.T,ys)

	C = linalg.inv(XX)

	w = dot(C,Xy)
	
	warnings.warn("Uncertainties estimates are uncertain! Require more testing.")
	
	if empirical_errors:
		
		system_spec = checkDimsBasis(x,w,expand)
		if system_spec <= 0:
			warnings.warn("Underdetermined or square system, empirical errors are meaningless.")
		
		warnings.warn("Using empirical errors, assuming that all errorbars are of similar value.")
		
		(p,_) = predict(x,w,C)
			
		std_diff = ones(p.shape)*std(p-y,ddof=1)
		
		# repeat with updated standard deviations
		(w,C) = fit(x,y,std_diff,empirical_errors=False)
		
		#pdb.set_trace()
	

	
	return (w,C)


def to_unit_box(x,invert_x):

	# min(x)=0
	# max(x)=1

	x_nobox = x.copy()
	min_x = np.min(x,axis=0)
	x = x-min_x

	max_x = np.max(x,axis=0)
	max_x_kron = np.kron(max_x,np.ones([x.shape[0],1]))

	x /= max_x_kron


	return x, min_x, max_x


def fit2(x,y,s="none",expand=linearBasis,empirical_errors=False,invert_x=False):

	(x,y,s) = checkDimsData(x,y,s)

	x_nobox = x.copy()
	x, min_x, max_x = to_unit_box(x,invert_x)

	X=expand(x)
	n_dims_X=X.shape[1]

	ys=y/s
	Xs=X/kron(ones((n_dims_X,1)),s).T

	XX = dot(Xs.T,Xs) + eps*identity(n_dims_X)
	Xy = dot(Xs.T,ys)

	C = linalg.inv(XX)

	w = dot(C,Xy)
	
	warnings.warn("Uncertainties estimates are uncertain! Require more testing.")
	
	if empirical_errors:
		
		system_spec = checkDimsBasis(x,w,expand)
		if system_spec <= 0:
			warnings.warn("Underdetermined or square system, empirical errors are meaningless.")
		
		warnings.warn("Using empirical errors, assuming that all errorbars are of similar value.")
		
		(p,_) = predict(x,w,C)
			
		std_diff = ones(p.shape)*std(p-y,ddof=1)
		
		# repeat with updated standard deviations
		(w,C) = fit(x,y,std_diff,empirical_errors=False)
		
		#pdb.set_trace()
	
	fitdata = {}	
	fitdata['min_x'] = min_x
	fitdata['max_x'] = max_x
	fitdata['invert_x'] = invert_x
	fitdata['expand'] = expand
	fitdata['w'] = w
	fitdata['C'] = C
	
	return fitdata

def predict2(x,fitdata):

	x_nobox = x.copy()

	expand = fitdata['expand']
	w = fitdata['w']
	C = fitdata['C']

	x=checkDimsX(x)
	checkDimsBasis(x,w,expand)

	x = x-fitdata['min_x']
	max_x_kron = np.kron(fitdata['max_x'],np.ones([x.shape[0],1]))
	x /= max_x_kron
	
	X=expand(x)
	n_dims_X = X.shape[1]
	n_dims_w = w.shape[0]
	n_points_x = x.shape[0]

	p = dot(X,w)
	s = zeros(n_points_x)
	
	for ip in range(n_points_x):
		
		iX = X[ip,:]
		XX= outer(iX,iX)
					
		s[ip] = sum(sum(XX*C))
		
	warnings.warn("Uncertainties estimates are uncertain! Require more testing.")
		

	return (p,s)


def predict(x,w,C,expand=linearBasis):

	x=checkDimsX(x)
	# checkDimsBasis(x,w,expand)
	
	X=expand(x)
	n_dims_X = X.shape[1]
	n_dims_w = w.shape[0]
	n_points_x = x.shape[0]



	p = dot(X,w)
	s = zeros(n_points_x)
	
	for ip in range(n_points_x):
		
		iX = X[ip,:]
		XX= outer(iX,iX)
					
		s[ip] = sum(sum(XX*C))
		
	warnings.warn("Uncertainties estimates are uncertain! Require more testing.")
		
	return (p,s)

def plotFit(train_x,train_y,train_s,w,C, n_points = 20, xdim=0):
		
	#plt.ion()
	plt.figure()
	plt.clf()
	
	(train_x,train_y,train_s) = checkDimsData(train_x,train_y,train_s)
			
	plt.errorbar(train_x[:,xdim],train_y,yerr=train_s,fmt='o')
	
	max_x = train_x[:,xdim].max() 
	min_x = train_x[:,xdim].min()   
	
	max_x += abs(max_x - min_x)*0.1
	min_x -= abs(max_x - min_x)*0.1
	
	test_x = linspace( min_x, max_x , n_points)[:,newaxis]
	
	(test_p,test_s) = predict(test_x,w,C)
	
	plt.plot(test_x[:,0],test_p)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()


	
def checkDimsData(x,y,s="none"):
	
	x=checkDimsX(x)
	
	if s=="none":
		s=ones(y.shape)

	if not (x.ndim == 1 or x.ndim == 2):
		warnings.warn("x.ndim = %d (2 required)" % x.ndim)
		return 0


		
	if y.ndim != 1:
		warnings.warn("y.ndim = %d (1 required)" % x.ndim) 
		return 0

	if s.ndim != 1:
		warnings.warn("s.ndim = %d (1 required)" % x.ndim)
		return 0


	n_points_x =x.shape[0]
	n_dims_x   =x.shape[1]
	n_points_y =y.shape[0]
	n_points_s =s.shape[0]

	if len(y.shape) != 1:

		warning.warn("len(y.shape) = %d (1 required)" % len(y.shape))
		return 0

	if len(s.shape) != 1:

		warnings.warn("len(s.shape) = %d (1 required)" % len(s.shape))
		return 0

	if n_points_x != n_points_y:

		warnings.warn("y.shape[0] = %d x.shape[0] = %d (should be equal)" % (y.shape[0],x.shape[0]))
		return 0

	if n_points_s != n_points_y:

		warnings.warn("y.shape[0] = %d s.shape[0] = %d (should be equal)" % (y.shape[0],s.shape[0]))
		return 0
	
	return (x,y,s)
	
def checkDimsBasis(x,w,expand=linearBasis):
	
	
	# use only the first one
	x=x[0,:]
	x=x[:,newaxis]
	X=expand(x)
	
	n_dims_X = X.shape[1]
	n_dims_w = w.shape[0]
	
	if n_dims_X != n_dims_w:

		warnings.warn("print X.shape[1] = %d w.shape[0] = %d (should be equal). Returning meaningless results." % (X.shape[1],w.shape[0]))
		return 666
	
	if n_dims_X > n_dims_w:
		return 1
	
	if n_dims_X == n_dims_w:
		return 0
	
	if n_dims_X < n_dims_w:
		return -1
	
def checkDimsX(x):
	
	if x.ndim == 1:
	#print "x.ndim = %d (2 required)" % x.ndim 
		x=x[:,newaxis]
	
	return x
	
	


def testLinear():

	n_points = 10;
	a = 3
	b = 2

	noise_std = 3

	test_x = linspace(-10,10,n_points)[:,newaxis]
	test_y = (a + test_x*b + random.randn(test_x.shape[0],1)*noise_std)[:,0]
	test_s = kron(ones(test_y.shape),noise_std)
	
	plt.ion()
	plt.clf()

	plt.errorbar(test_x[:,0],test_y,yerr=test_s)
	plt.plot(test_x[:,0],a+test_x[:,0]*b)


	# return
	(w,C)=fit(x=test_x,y=test_y,s=test_s)
	
	print "C"
	print C

	print "w"
	print w
	
	
	(test_p,test_s) = predict(test_x,w,C)
	
	# print "test_p"
	# print test_p
	# print "test_s"
	# print test_s 
	
	plt.plot(test_x[:,0],a+test_x[:,0]*b)
	plt.plot(test_x[:,0],test_p)

	savetxt('test_x.txt',test_x);
	savetxt('test_y.txt',test_y);
	savetxt('test_s.txt',test_s);

def get_line_fit(x,y,sig):
        """
        @brief get linear least squares fit with uncertainity estimates
        y(X) = b*X + a
        see numerical recipies 15.2.9
        @param X    function arguments 
        @param y    function values
        @param sig  function values standard deviations   
        @return a - additive 
        @return b - multiplicative
        @return C - covariance matrix
        """
        
        invsig2 = sig**-2;
        
        S  = np.sum(invsig2)
        Sx = np.inner(x,invsig2)
        Sy = np.inner(y,invsig2)
        Sxx = np.inner(invsig2*x,x)
        Sxy = np.inner(invsig2*x,y)

        D = S*Sxx - Sx**2
        a = (Sxx*Sy - Sx*Sxy)/D
        b = (S*Sxy  - Sx*Sy)/D
        
        Cab = np.zeros((2,2))
        Cab[0,0] = Sxx/D
        Cab[1,1] = S/D
        Cab[1,0] = -Sx/D
        Cab[0,1] = Cab[1,0]
        
        return a,b,Cab





if __name__ == "__main__":
    testLinear()


