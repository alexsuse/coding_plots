import numpy as np

def factorial(n):
	temp = 1
	for i in range(n):
		temp*=(i+1)
	return temp

def binomial(n,k):
	return factorial(n)/(factorial(k)*factorial(n-k))

class GaussianEnv(object):
	"""Implements a gaussian process with correlation structure
	<S_i(t_k)S_j(t_l)> = exp(-(|x(i)-x(j)|/L)**zeta)*phi(|t_k-t_l|;eta,gamma,order)
	we take the points x to be regularly distributed on a grid on the square
	(x0,y0),(x0,y0+Ly),(x0+Lx,y0+Ly),(x0+Lx,y0) and phi is the matern kernel
	with \nu = order-1/2."""
	def __init__(self,zeta,gamma,eta,L,N,x0,y0,Lx,Ly,sigma,order):
		"""Constructer method, generates all the internal data
		xs and ys are the x and y coordinates of all the points.
		zeta, eta, gamma, L are the parameters of the kernel
		N is the number of subdivisions in the grid, sigma is added
		to the diagonal of the gram matrix to ensure positive definiteness
		and order is the order of the matern process
		k is the covariance matrix, and khalf its cholesky decomposition, used to sample"""
		self.xs = np.arange(x0,x0+Lx,Lx/N)
		self.ys = np.arange(y0,y0+Ly,Ly/N)
		self.zeta = zeta
		self.gamma = gamma
		self.eta = eta
		self.L = L
		self.N = N
		self.k = np.zeros((N*N,N*N))
		self.S = np.random.normal(0.0,1.0,(order,N*N))
		self.order = order if order>0 else 1
		if N>1:
			for i in range(0,N*N):
				for j in range(0,N*N):
					(ix,iy) =divmod(i,N)
					(jx,jy) =divmod(j,N)
					dist = np.sqrt((self.xs[ix]-self.xs[jx])**2 + (self.ys[iy]-self.ys[jy])**2) 
					self.k[i,j] = np.exp(-(dist/L)**zeta)+sigma*(i==j)
			self.khalf = np.linalg.cholesky(self.k)
		else:
			self.khalf = np.array([1.0])
	def reset(self):
		self.S = np.random.normal(0.0,1.0,(self.order,self.N*self.N))

	def getgamma(self):
		g = np.zeros((self.order,self.order))
		for i in range(self.order):
			g[i-1,i] = -1.0
		for i in range(self.order):
			g[-1,i] = binomial(self.order,i)*self.gamma**(self.order-i)
		return g		

	def geteta(self):
		eta = np.zeros((self.order,self.order))
		eta[-1,-1]= self.eta**2
		return eta

	def sample(self):
		"""Gets an independent sample from the spatial kernel"""
		s = np.random.normal(0.0,1.0,self.N*self.N)
		s = np.dot(self.khalf,s)
		return np.reshape(s,(self.N,self.N))
		
	def samplestep(self,dt,N =1 ):
		"""Gives a sample of the temporal dependent gp"""
		sample = np.zeros((N,self.order))
		for steps in range(N):
			temp = 0.
			for j in range(0,self.order):
				temp = temp+binomial(self.order,j)*self.gamma**(self.order-j)*self.S[j,:]
			self.S[-1,:]= self.S[-1,:]-dt*temp+np.sqrt(dt)*self.eta*np.random.normal(0.0,1.0,self.N*self.N)
			for i in reversed(range(self.order-1)):
				self.S[i,:] = self.S[i,:] + dt*self.S[i+1,:]
			sample[steps,:] = self.S[:,0]
		#sample = np.dot(self.khalf,self.S[-1,:])
		return sample
		#return np.reshape(sample,(self.N,self.N))[::-1]
		
		
