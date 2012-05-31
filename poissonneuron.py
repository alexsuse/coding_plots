import numpy as np
import random

class IFNeuron(object):
	"""Implements a leaky integrate and fire neuron"""
	def __init__(self,tau,k=None,thresh=1.0,reset=0.0,stochastic=False, alpha=0.0):
		self.tau = tau
		self.V = 0
		self.stochastic = stochastic
		self.alpha = alpha
		if self.stochastic:
			self.averagethresh = thresh
			self.thresh = np.random.normal(self.averagethresh,self.alpha,1)
		else:
			self.thresh = thresh
		self.reset = reset
		if k == None:
			self.k = np.random.normal((1,))
		else:
			self.k = k
		
	def spike(self,stim,dt):
		self.V = (1-dt/self.tau)*self.V + dt*self.modulation(stim)
		if self.V > self.thresh:
			if self.stochastic:
				self.thresh = np.random.normal(self.averagethresh,self.alpha,1)
			self.V = 0
			return 1
		return 0
	def modulation(self,stim):
		return np.dot(self.k,stim)
		

class PoissonPlasticNeuron(object):
	"""Implements a poisson neuron with a gaussian tuning function"""
	def __init__(self,theta,A,phi,tau,N=None):
		"""Generates the neuron object,
		X are the coordinates of the stimulus points
		A is the correlation of the gaussian tuning function
		phi is the maximal firing rate
		inva the inverse of a is calculated once for reuse"""
		
		self.A=A
		if N == None:
			N = np.size(theta)
		else:
			self.N = N
		self.inva = np.array(np.matrix(A).I)
		self.phi=phi
		self.theta = theta
		self.mu = 1.0
		self.dm = 0.05
		self.tau = tau
	def rate(self,stim):
		"""Gives the rate of the poisson spiking process given the stimulus S"""
		#S = np.reshape(stim,self.N*self.N)
		S = stim
		exponent = np.dot(S.transpose()-self.theta.transpose(),np.dot(self.inva,S-self.theta))
		return self.phi*self.mu*np.exp(-0.5*exponent)
	def spike(self,S,dt):
		"""Generates a spike with probability rate*dt"""
		r = dt*self.rate(S)
		if np.random.uniform()<r:
			self.mu = self.mu - self.dm
			self.mu = self.mu if self.mu > 0.0 else 0.0
			return 1
		self.mu = self.mu+(1-self.mu)*dt/self.tau
		return 0

def choice(p,a=None,shape=(1,)):
	"""chooses an element from a with probabilities p. Can return arbitrarily shaped-samples through the shape argument.
	p needs not be normalized, as this is checked for."""
	x = np.random.uniform(size=shape)
	cump = np.cumsum(p)
	if cump[-1]!=1:
		cump=cump/cump[-1]
	idxs = np.searchsorted(cump,x)
	if a==None:
		return idxs
	else:
		return a[idxs]

class PoissonRessonantNeuron(object):
	"""Implements a poisson neuron with a gaussian tuning function and resonant adaptation"""
	def __init__(self,theta,A,phi,tau,omega,N=None):
		"""Generates the neuron object,
		X are the coordinates of the stimulus points
		A is the correlation of the gaussian tuning function
		phi is the maximal firing rate
		inva the inverse of a is calculated once for reuse"""
		
		self.A=A
		if N == None:
			N = np.size(theta)
		else:
			self.N = N
		self.inva = np.array(np.matrix(A).I)
		self.phi=phi
		self.theta = theta
		self.mu1 = 1.0
		self.mu2 = 0.0
		self.omega = omega
		self.dm = 0.05
		self.tau = tau
	def rate(self,stim):
		"""Gives the rate of the poisson spiking process given the stimulus S"""
		#S = np.reshape(stim,self.N*self.N)
		S = stim
		exponent = np.dot(S.transpose()-self.theta.transpose(),np.dot(self.inva,S-self.theta))
		return self.phi*self.mu1*np.exp(-0.5*exponent)
	def spike(self,S,dt):
		"""Generates a spike with probability rate*dt"""
		r = dt*self.rate(S)
		self.mu1 = self.mu1+ dt*self.mu2
		self.mu2 =  self.mu2+dt*(self.omega**2*(1.0-self.mu1) - self.mu2/self.tau)
		if np.random.uniform()<r:
			self.mu1 = self.mu1 - self.dm
			return 1
		return 0

def choice(p,a=None,shape=(1,)):
	"""chooses an element from a with probabilities p. Can return arbitrarily shaped-samples through the shape argument.
	p needs not be normalized, as this is checked for."""
	x = np.random.uniform(size=shape)
	cump = np.cumsum(p)
	if cump[-1]!=1:
		cump=cump/cump[-1]
	idxs = np.searchsorted(cump,x)
	if a==None:
		return idxs
	else:
		return a[idxs]

class PoissonNeuron(object):
	"""Implements a poisson neuron with a gaussian tuning function"""
	def __init__(self,theta,A,phi,N):
		"""Generates the neuron object,
		X are the coordinates of the stimulus points
		A is the correlation of the gaussian tuning function
		phi is the maximal firing rate
		inva the inverse of a is calculated once for reuse"""
		self.A=A**2
		self.N = N
		self.inva = np.array(np.matrix(A**2).I)
		self.phi=phi
		self.theta = np.array([theta])
	def rate(self,stim):
		"""Gives the rate of the poisson spiking process given the stimulus S"""
		S = np.reshape(stim,self.N*self.N)
		exponent = np.dot(S.transpose()-self.theta.transpose(),np.dot(self.inva,S-self.theta))
		return self.phi*np.exp(-0.5*exponent)
	def spike(self,S,dt):
		"""Generates a spike with probability rate*dt"""
		r = dt*self.rate(S)
		if np.random.uniform()<r:
			return 1
		return 0


class PoissonCode(object):
	def __init__(self,thetas,A,phi):
		self.neurons = []
		self.thetas = thetas
		self.N = thetas[0].size
		for theta in thetas:
			self.neurons.append(PoissonNeuron(theta,A,phi,self.N))
	def rates(self,stim):
		rs = []
		for n in self.neurons:
			rs.append(n.rate(stim))
		return rs
	def totalrate(self,stim):
		return sum(self.rates(stim))
	def spikes(self,stim,dt):
		sps = np.zeros_like(self.neurons)
		rates = self.rates(stim)
		rate = np.sum(rates)*dt
		if np.random.uniform()<rate:
			spiker = choice(rates)
			sps[spiker]=1
		return sps

class PoissonPlasticCode(object):
	def __init__(self,thetas=np.arange(-2.0,2.0,0.1),A = None, phi=1,N=40,alpha=1.0):
		self.N = np.size(thetas)
		self.A = A
		if A == None:
			self.A= alpha*np.eye(np.size(thetas[0]))
		self.neurons = []
		for theta in thetas:
			self.neurons.append(PoissonPlasticNeuron(theta,self.A,phi,N))
	def rates(self,stim):
		rs = []
		for n in self.neurons:
			rs.append(n.rate(stim))
		return rs
	def totalrate(self,stim):
		return np.sum(self.rates(stim))
	def spikes(self,stim,dt):
		r = self.rates(stim)
		tot = np.sum(r)
		sps = np.zeros(self.N)
		if np.random.uniform() < tot*dt:
			neu = choice(p=r)
			sps[neu]=1
		return sps
