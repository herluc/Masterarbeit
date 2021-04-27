import sys
from fenics import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.stats import norm
#



class solver:
	def __init__(self):
		self.dom_a = 0.0 #domain boundaries
		self.dom_b = 1.0

		self.ne	= 111 #number of elements
		self.nMC = 2000 #number of Monte Carlo points

		self.mesh = IntervalMesh(self.ne,self.dom_a,self.dom_b) #define mesh
		self.bbt = self.mesh.bounding_box_tree()


		self.X = np.expand_dims(np.linspace(self.dom_a, self.dom_b, self.ne), 1) # Vector of poitns in the mesh
		self.coordinates = self.mesh.coordinates() # vertices vector

		self.lf = 0.25
		self.sigf = 0.3

		#mean for f:
		self.mean = (np.pi)**2*(1/5)*np.ones(self.ne)
		#self.mean = (1.0)*np.ones(self.ne)


	def exponentiated_quadratic(self, xa, xb, lf, sigf):
		"""Exponentiated quadratic  with σ=1"""
		# L2 distance (Squared Euclidian)
		sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
		return sigf**2 * np.exp(sq_norm)


	
	def draw_FEM_samples(self, x_vals):
		
		print(self.mean)
		cov = self.exponentiated_quadratic(x_vals, x_vals, lf=self.lf, sigf=self.sigf)  # Kernel of data points
		discrete_f = np.random.multivariate_normal(
			mean = self.mean , cov=cov,
			size=1)
		return discrete_f



	def doFEM(self):
		# Define Dirichlet boundary (x = 0 or x = 1)
		def boundary(x):
			return x[0] < (self.dom_a + DOLFIN_EPS) or x[0] > self.dom_b - DOLFIN_EPS

		self.V = FunctionSpace(self.mesh, "Lagrange", 1) # Function space

		#get element barycentres:
		coordinates = (self.coordinates[1:] + self.coordinates[:-1])/2


		# Define boundary condition
		u0 = Constant(0.0)
		self.bc = DirichletBC(self.V, u0, boundary)

		# Define variational problem
		self.u = TrialFunction(self.V)
		self.v = TestFunction(self.V)

		bbt = self.mesh.bounding_box_tree()
		class DoGP(UserExpression):
			def eval(self, value, x):
				collisions1st = bbt.compute_first_entity_collision(Point(x[0])) # find index of cell which contains point.
				value[0] = f_vals[collisions1st] # f is constant in a cell. therefore there are 100 cells and 100 values for f.
			def value_shape(self):
				return ()
		f_vals = self.draw_FEM_samples(coordinates)
		f_vals = f_vals[0]
		f = DoGP()


		a = dot(grad(self.u), grad(self.v))*dx #variational Problem
		L = f*self.v*dx

		self.u = Function(self.V)
		A = assemble(a)
		b = assemble(L)
		self.bc.apply(A, b)
		self.u = Function(self.V)
		U = self.u.vector()
		solve(A, U, b)
		return U, A



	def get_U_mean(self):
		"""
		calculates the mean for the FEM solution prior GP
		"""
		u_mean = Function(self.V)
		U_mean = u_mean.vector()
		b_mean = assemble(self.mean[1]*self.v*dx) # because for the mean, f = 1. (1 * v * dx)
		self.bc.apply(b_mean)
		solstd,A = self.doFEM()  # solstd is unused here.
		solve(A,U_mean,b_mean)
		return U_mean


	def get_U_mean_verbose(self):
		
		rhs = assemble(self.mean[1] * self.v * dx).get_local()
		rhs[0]=0.0 #Boundary conditions
		rhs[-1]=0.0
		ident = np.identity(len(self.coordinates))
		solstd,A = self.doFEM()
		A_inv = np.linalg.solve(A.array(), ident)
		U_mean_verbose = A_inv.dot(rhs)
		return U_mean_verbose


	
	def get_C_f(self):
		#integratedTestF = np.array(assemble(Constant(1.) * self.v * dx).get_local()) #* self.ne
		#self.integratedTestF = np.array(assemble(Constant(self.mean[1]) * self.v * dx).get_local()) #* self.ne
		self.integratedTestF = np.array(assemble(Constant(1.0) * self.v * dx).get_local()) #* self.ne

		print(self.integratedTestF)
		c_f = self.exponentiated_quadratic(self.coordinates,
			 self.coordinates, lf=self.lf, sigf=self.sigf)
		C_f = np.zeros((self.ne+1,self.ne+1))
		for i in range(self.ne+1):
			for j in range(self.ne+1):
				C_f[i,j] = self.integratedTestF[i] * c_f[i,j] * self.integratedTestF[j]
		return C_f



	def get_C_u(self):
		#integratedTestF = np.array(assemble(Constant(1.) * self.v * dx).get_local())
		#print(integratedTestF)
		#c_f = self.exponentiated_quadratic(self.coordinates,
		#	 self.coordinates, lf=self.lf, sigf=self.sigf)
		#C_f = (self.integratedTestF) * c_f * (self.integratedTestF)
		C_f = self.get_C_f()
		solstd,A = self.doFEM()
		ident = np.identity(len(self.coordinates))
		self.bc.apply(A)
		A = A.array()
		print("A")
		print(A)
		print("Determinant of A: ",np.linalg.det(A))
		#A_inv = np.linalg.inv(A)
		A_inv = np.linalg.solve(A,ident)
		thresh = 1e-16
		#ind_smaller_thresh = A_inv < thresh
		#A_inv[ind_smaller_thresh] = 0
		print("A_inv")
		print(A_inv)
		#C_u = A_inv * C_f * np.transpose(A_inv)
		C_u = np.dot( np.dot(A_inv,C_f), np.transpose(A_inv))
		#C_u = C_u + 0.000001*(self.sigf**2)*ident
		c_u = np.transpose(self.integratedTestF) * C_u * self.integratedTestF
		print("A^⁻T")
		print(np.transpose(A_inv))
		print("C_u")
		print(C_u)
		return C_u
		#return c_u



	def samplePrior(self):
		U_mean = self.get_U_mean()
		C_u = self.get_C_u()

		priorGP = np.random.multivariate_normal(
			mean = U_mean, cov=C_u,
			size=self.nMC)

		return priorGP


	def doMC(self,samples):
		gamma = 0.95
		n = len(samples)
		#print('samples:')
		#print(n)
		#print(samples)
		mu = np.sum(samples) / n
		var = (1/(n-1)) * np.sum( np.square(samples-mu) )
		sig = np.sqrt(var)
		d = -norm.ppf((1-gamma)/2)
		Delta = sig
		return (mu, sig, Delta)




solver = solver()
U,A = solver.doFEM()
U_mean = solver.get_U_mean()
U_mean_verbose = solver.get_U_mean_verbose()
#C_u = solver.get_C_u()
solver.get_C_f()
priorSamples = solver.samplePrior()
print(np.transpose(priorSamples))

sigL =[]
muL = []
DeltaL = []
for i in range(solver.ne+1):
	mu,sig,Delta = solver.doMC(np.transpose(priorSamples)[i])
	sigL.append(sig)
	muL.append(mu)
	DeltaL.append(Delta)
#print(solver.coordinates)
print(sigL)
print(np.transpose(solver.coordinates)[0])
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(solver.coordinates, np.transpose(U_mean), linestyle='-', color = 'black',lw = 1.0,label='Mean analyt.')
#plt.plot(solver.coordinates, np.transpose(U_mean_verbose), linestyle='-.', color = 'red',lw = 1.0)
plt.plot(solver.coordinates, np.transpose(priorSamples[110:120]), linestyle='-')
plt.plot(np.transpose(solver.coordinates)[0], np.array(muL), linestyle='-.',color = 'black',lw = 3.0, label='Mean MC')
plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)-2*np.array(sigL), linestyle='-',color = 'black',lw = 3.0,label='2sig')
plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)+2*np.array(sigL), linestyle='-',color = 'black',lw = 3.0)
#plt.fill_between((solver.coordinates)[0], np.array(muL)-np.array(DeltaL), np.array(muL)+np.array(DeltaL),color = 'blue',label='2sig')
plt.legend()
plt.xlim([solver.dom_a, solver.dom_b])

plt.show()
#plt.pause(0.01) # Pause for interval seconds.
#input("hit[enter] to end.")
#plt.close('all') # all open plots are correctly closed after each run













