import sys
from fenics import *
import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')
import time
import seaborn as sns
import numpy as np
#np.seterr(all='ignore')
import scipy
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
#
#from numba import jit


class solverClass:
	def __init__(self, nMC = 200):
		self.dom_a = 0.0 #domain boundaries
		self.dom_b = 1.0

		self.ne	= 30 #number of elements
		self.nMC = nMC #number of Monte Carlo points

		self.mesh = IntervalMesh(self.ne,self.dom_a,self.dom_b) #define mesh
		self.bbt = self.mesh.bounding_box_tree()


		self.X = np.expand_dims(np.linspace(self.dom_a, self.dom_b, self.ne), 1) # Vector of poitns in the mesh
		self.coordinates = self.mesh.coordinates() # vertices vector

		self.lf = 0.25
		self.sigf = 0.3

		#mean for f:
		self.mean = (np.pi)**2*(1/5)*np.ones(self.ne)
		#self.mean = (1.0)*np.ones(self.ne)
		self.V = FunctionSpace(self.mesh, "Lagrange", 1) # Function space
		self.el = self.V.element()


	def exponentiated_quadratic(self, xa, xb, lf, sigf):
		"""Exponentiated quadratic  with σ=1"""
		# L2 distance (Squared Euclidian)
		sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
		return sigf**2 * np.exp(sq_norm)



	def exponentiated_quadratic_log(self, xa, xb, l, sig):
		""" expects input parameters as log(par). """
		return np.exp(2*sig) * np.exp(-0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * np.exp(-2*l))

	def matern_log(self, xa, xb, l, sig):
		""" expects input parameters as log(par). """
		r = scipy.spatial.distance.cdist(xa, xb, 'euclidean')
		return np.exp(2*sig) * np.exp(-1 * scipy.spatial.distance.cdist(xa, xb, 'euclidean') * np.exp(-1*l)) #nu = 1/2
		#return np.exp(2*sig) * ((1 + np.sqrt(3) * r *np.exp(-1*l)) *  np.exp(-1 * np.sqrt(3) * r * np.exp(-1*l))   ) # nu = 3/2
		#return np.exp(2*sig) * ((1 + np.sqrt(5) * r *np.exp(-1*l) +  5*r*r/3*np.exp(-1*l)*np.exp(-1*l)  ) *  np.exp(-1 * np.sqrt(5) * r * np.exp(-1*l))   ) # nu = 5/2



	def draw_FEM_samples(self, x_vals):
		#print("fem mean f")
		#print(self.mean)
		cov = self.exponentiated_quadratic(x_vals, x_vals, lf=self.lf, sigf=self.sigf)  # Kernel of data points
		discrete_f = np.random.multivariate_normal(
			mean = self.mean , cov=cov,
			size=1)
		return discrete_f



	def create_fake_data(self, y_points,y_values):
		#parameters:
		rho = 1.2
		sig_d = 2e-2
		l_d = 0.5
		sig_y = 5e-3
		ny = len(y_points)
		print(np.shape(y_points))
		#print(ny)
		y_points = np.array(y_points)
		#print(y_points)
		y1 = (np.sin(np.pi*y_points)/3 * np.sin(7*np.pi*y_points)/40)
		y2 = np.transpose(np.atleast_2d(np.random.multivariate_normal(mean = np.zeros(ny), cov = self.exponentiated_quadratic(y_points,y_points,l_d,sig_d))))
		y3 = np.transpose(np.atleast_2d(np.random.normal(scale = sig_y, size = ny)))
		#print(np.shape(np.transpose(np.atleast_2d(np.array(self.y_values)))))
		y = y1 +rho *np.transpose(np.atleast_2d(np.array(y_values)))

		print(y.flatten())
		plt.figure(figsize=(6, 4), dpi=100)
		plt.plot(y_points,y)
		plt.show()
		return y.flatten()



	def doFEM(self):
		# Define Dirichlet boundary (x = 0 or x = 1)
		def boundary(x):
			return x[0] < (self.dom_a + DOLFIN_EPS) or x[0] > self.dom_b - DOLFIN_EPS

		#self.V = FunctionSpace(self.mesh, "Lagrange", 1) # Function space

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
		#print(self.mean)
		#print(self.mean[1])
		b_mean = assemble(self.mean[1]*self.v*dx) # because for the mean, f = 1. (1 * v * dx)
		self.bc.apply(b_mean)
		solstd,A = self.doFEM()  # solstd is unused here.
		solve(A,U_mean,b_mean)
		self.U_mean = U_mean
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
		self.integratedTestF = np.array(assemble(Constant(self.mean[1]) * self.v * dx).get_local()) #* self.ne

		self.integratedTestF = np.array(assemble(Constant(1.0) * self.v * dx).get_local()) #* self.ne

		print(self.integratedTestF)
		# c_f = self.exponentiated_quadratic(self.coordinates,
		# 	 self.coordinates, lf=self.lf, sigf=self.sigf)

		c_f = self.exponentiated_quadratic_log(self.coordinates,
			 self.coordinates, l=np.log(self.lf), sig=np.log(self.sigf))

		c_f = self.matern_log(self.coordinates,
			 self.coordinates, l=np.log(self.lf), sig=np.log(self.sigf))
		self.c_f = c_f
		C_f = np.zeros((self.ne+1,self.ne+1))
		#print('coords:')
		#print(self.coordinates)
		for i in range(self.ne+1):
			for j in range(self.ne+1):
				C_f[i,j] = self.integratedTestF[i] * c_f[i,j] * self.integratedTestF[j]
		#self.bc.apply(C_f)




		return C_f



	def get_C_u(self):
		C_f = self.get_C_f()
		solstd,A = self.doFEM()
		ident = np.identity(len(self.coordinates))
		self.bc.apply(A)
		A = A.array()

		#A_inv = np.linalg.inv(A)  ##needs to be choleskied!
		A_inv = np.linalg.solve(A,ident)
		C_u = np.dot( np.dot(A_inv,C_f), np.transpose(A_inv))
		self.C_uDiag = np.sqrt(np.diagonal(C_u))
		condBefore = np.linalg.cond(C_u)
		C_u = C_u + 1e-10*(self.sigf**2)*ident
		condAfter = np.linalg.cond(C_u)
		c_u = np.transpose(self.integratedTestF) * C_u * self.integratedTestF
		self.C_u = C_u
		print("Cond:")
		print(condBefore,condAfter)
		#self.C_uDiag = np.sqrt(np.diagonal(self.C_u))

		return C_u
		#return c_u



	def samplePrior(self):
		U_mean = self.get_U_mean()
		C_u = self.get_C_u()
		priorGP = np.random.multivariate_normal(
			mean = U_mean, cov=C_u,
			size=self.nMC)

		return priorGP


	def samplef(self):
		f_mean = (np.pi)**2*(1/5)*np.ones(self.ne+1)
		fGP = np.random.multivariate_normal(
			mean = f_mean, cov=self.c_f,
			size=10)
		f = plt.figure(figsize=(6, 4), dpi=100)
		plt.plot(self.coordinates, np.transpose(f_mean), linestyle='-', color = 'black',lw = 1.0)
		#plt.plot(solver.coordinates, np.transpose(U_mean_verbose), linestyle='-.', color = 'red',lw = 1.0)
		plt.plot(self.coordinates, np.transpose(fGP), linestyle='-',lw = 0.4)
		C_fDiag = np.sqrt(np.diagonal(self.c_f))
		plt.plot(np.transpose(solver.coordinates)[0], np.transpose(f_mean)-1.96*C_fDiag, linestyle='-.',color = 'green',lw = 1.0,label='2sig')
		plt.plot(np.transpose(solver.coordinates)[0], np.transpose(f_mean)+1.96*C_fDiag, linestyle='-.',color = 'green',lw = 1.0)
		plt.ylabel("$f(x)$")
		plt.xlabel("$x$")
		plt.grid()
		#plt.legend()
		f.savefig("matern1_2_f_sampled.pdf", bbox_inches='tight')
		#f.savefig("sqex_f_sampled.pdf", bbox_inches='tight')
		plt.show()


	def doMC(self,samples):
		gamma = 0.95
		n = len(samples)
		mu = np.sum(samples) / n
		var = (1/(n-1)) * np.sum( np.square(samples-mu) )
		sig = np.sqrt(var)
		d = -norm.ppf((1-gamma)/2)
		Delta = sig

		return (mu, sig, Delta)




	def doMCMC(self,w0):
		sig_q = 0.002
		cov = sig_q**2 * np.identity(len(w0))
		numSamp = 500
		samples = []
		#w0 =[1,1,1]
		samples.append(w0)
		wi = w0

		for i in list(range(numSamp))[1:]:
			#proposal =  samples[i-1] + np.random.normal(loc=0.0, scale = sig_q, size=len(w0))
			proposal = np.random.multivariate_normal(
			mean = wi, cov=cov)
			#proposal = abs(proposal)
			# print("prop:")
			# print(proposal)
			# print(self.getLogPost(proposal))
			# print(self.getLogPost(wi))
			# print(self.getLogPost(proposal)  -  self.getLogPost(wi) )
			alpha = np.min(     (0.0, self.getLogPostMultiple(proposal)  -  self.getLogPostMultiple(wi) )  )
			#print("alpha: ",alpha)
			u = np.random.uniform(0,1)
			if min(proposal)>0:
				if np.log(u) < alpha:
					samples.append(proposal)
					wi = proposal
				else:
					samples.append(wi)
					wi = wi
			else:
				samples.append(wi)
				wi = wi

		quarter = int(len(samples)/4)
		mean = np.mean(samples[quarter:],axis=0)
		#plt.figure(figsize=(6, 4), dpi=100)
		#plt.plot(np.transpose(samples)[0],label=['rho','sig',"l"])
		#plt.yscale('log')
		#plt.xscale('log')

		#plt.hist(np.transpose(samples),bins=30)
#		plt.grid()
	#	plt.legend()


		#print("mean:")
		#print(mean)
		return mean#np.array(samples)




	def getP(self,y_points):
		y_points = y_points#[0.4,0.933333]
		ny = len(y_points)
		P = np.zeros((ny, self.ne+1), dtype = float)
		#x = np.array([0.334])

		for j,point in enumerate(y_points):
			x = np.array([point])

			x_point = Point(x)
			cell_id = self.mesh.bounding_box_tree().compute_first_entity_collision(x_point)
			cell = Cell(self.mesh, cell_id)
			#print('cell id:')
			#print(cell_id)
			coordinate_dofs = cell.get_vertex_coordinates()
			#print('coord dofs:')
			#print(coordinate_dofs)
			values = np.ones(1, dtype=float)
			for i in range(self.el.space_dimension()):
				phi_x = self.el.evaluate_basis(np.array(i), x ,np.array(coordinate_dofs),cell.orientation())
				P[j,cell_id+i] = phi_x
		self.Pu = np.dot(P,self.U_mean)
		#print('Pu:')
		#print(np.dot(P,self.U_mean))
		self.P = P
		self.P_T = np.transpose(P)
		return P



	def get_C_d(self,y_points,ld, sigd):
		#ld = 1e-16
		#sigd =1e-16
		#C_d = self.exponentiated_quadratic(y_points, y_points, lf=ld, sigf=sigd)
		C_d = self.exponentiated_quadratic_log(y_points, y_points, l=ld, sig=sigd)
		C_d = self.matern_log(y_points, y_points, l=ld, sig=sigd)
		#print('C_d:')
		#print(C_d)
		self.C_d = C_d

		return C_d


	def get_C_d_derivSigd(self,y_points,ld, sigd):
		sq_norm = -0.5 * scipy.spatial.distance.cdist(y_points, y_points, 'sqeuclidean') * (1/ld**2)
		return 2*sigd* np.exp(sq_norm)

	def get_C_d_derivld(self,y_points,ld, sigd):
		sq_norm = -0.5 * scipy.spatial.distance.cdist(y_points, y_points, 'sqeuclidean') * (1/ld**2)
		return sigd**2 * np.exp(sq_norm) * 1/(ld**3) * scipy.spatial.distance.cdist(y_points, y_points, 'sqeuclidean')



	def get_C_e(self,size):
		sige_square = 2.5e-5
		C_e = sige_square * np.identity(size)
		#C_e = np.zeros((size,size))
		self.C_e = C_e
		#print('C_e:')
		#print(C_e)
		return C_e


	def getLogPost(self,params):
		rho = params[0]
		sigd = params[1]
		ld = params[2]
		#print("params:")
		#print(rho,sigd,ld)
		y_points = np.transpose(np.atleast_2d(np.array(self.y_points)))
		y_values = np.array(self.y_values)

		rho = np.exp(rho)
		C_u_trans = np.dot(   np.dot(self.P , self.C_u )  ,self.P_T   )
		#C_u_trans = np.dot(self.P, np.dot(self.C_u,self.P_T))
		#print(self.P)
		#print(self.P_T)
		K_y = self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ self.C_e + rho * rho *  C_u_trans
		#K_y = rho * rho *  C_u_trans


		# K_y_inv = np.linalg.inv(K_y)
		# ny = len(y_values)
		# y = rho * self.Pu - y_values
		# K_y_det = np.linalg.det(K_y)
		# logpost = 0.5 * np.dot(np.transpose(y), np.dot(K_y_inv , y)  )  +0.5*np.log(K_y_det) + ny/2 * np.log(2* np.pi) #Version Paper
		# #print(K_y @ np.dot(K_y_inv , y) - y)

		ny = len(y_values)
		y = y_values - rho * self.Pu

		L = cho_factor(K_y)
		K_y_inv_y = cho_solve(L, y)
		Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
		logpost2 = 0.5 * (np.dot(np.transpose(y), K_y_inv_y )  +Log_K_y_det + ny * np.log(2* np.pi))#Version Paper
		#print(K_y @ K_y_inv_y - y)




		#logpost = 0.5 * np.dot(np.transpose(y), np.dot(K_y_inv , y)  )  +0.5*np.log(K_y_det) + ny/2 * np.log(2* np.pi) #Version Paper

		#logpost = ny * np.pi + 0.5*K_y_det + 0.5* np.dot(np.transpose(y), np.dot(K_y_inv, y) ) #Version Römer
		return logpost2


	def getLogPostMultiple(self,params):
		rho = params[0]
		sigd = params[1]
		ld = params[2]
		y_valuesList=self.yVectors
		y_points = np.transpose(np.atleast_2d(np.array(self.y_points)))
		logpost = 0
		rho = np.exp(rho)
		C_u_trans = np.dot(   np.dot(self.P , self.C_u )  ,self.P_T   )

		K_y = self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ self.C_e + rho * rho *  C_u_trans
		L = cho_factor(K_y)
		for i,obs in enumerate(y_valuesList):
			y_values = np.array(obs)

			ny = len(y_values)
			y = y_values - rho * self.Pu

			K_y_inv_y = cho_solve(L, y)
			Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
			logpost2 = 0.5 * (np.dot(np.transpose(y), K_y_inv_y )  +Log_K_y_det + ny * np.log(2* np.pi))#Version Paper
			logpost = logpost + logpost2
			#print(i)


		return logpost



	def getLogPostDeriv(self,params):
		rho = params[0]
		sigd = params[1]
		ld = params[2]

		y_points = np.transpose(np.atleast_2d(np.array(self.y_points)))
		y_values = np.array(self.y_values)
		C_u_trans = np.dot(self.P, np.dot(self.C_u,self.P_T))
		K = rho * rho * C_u_trans + self.get_C_d(y_points = y_points,ld=ld,sigd=sigd) + self.C_e
		K_inv = np.linalg.inv(K)
		K_det = np.linalg.det(K)
		rho = np.exp(rho)  ##wieso?!
		y = rho * self.Pu - y_values
		ny = len(y_values)

		dKdrho = 2 * rho * C_u_trans
		dKdsigd = self.get_C_d_derivSigd(y_points,ld,sigd)
		dKdld = self.get_C_d_derivld(y_points,ld,sigd)
		alpha = np.dot(K_inv, y)
		alphaT = np.transpose(alpha)

		logpostDerivRho = -0.5 * np.trace(  np.dot((np.dot(alpha,alphaT)  - K_inv) ,   dKdrho        )  )
		logpostDerivSigd = -0.5 * np.trace(  np.dot((np.dot(alpha,alphaT)  - K_inv) ,   dKdsigd        )  )
		logpostDerivLd = -0.5 * np.trace(  np.dot((np.dot(alpha,alphaT)  - K_inv) ,   dKdld      )  )


		return np.array([logpostDerivRho,logpostDerivSigd,logpostDerivLd])



	def estimateHyperpar(self,y_points,y_values):
		#ld = 0.01
		#sigd = 0.1
		#rho = 0.85
		y_points = np.transpose(np.atleast_2d(np.array(y_points)))
		y_values = np.array(y_values)
		logpostList = []

		ld_s = np.log(np.logspace(-5,0.0001,5000,base=np.exp(1)))
		sigd_s = np.log(np.logspace(-8,0.8,5000,base=np.exp(1)))
		rho_s = np.log(np.logspace(-0.8,1,5000,base=np.exp(1)))
		#print("l,sig,rho:")
		#print(ld_s)
		#print(sigd_s)
		#print(rho_s)

		#rho_s = np.log(np.random.uniform(0.5,1.5,10000))
		#ld_s = np.log(np.random.uniform(1e-16,1,10000))
		#sigd_s = np.log(np.random.uniform(1e-16,1,10000))

		np.random.shuffle(ld_s)
		np.random.shuffle(sigd_s)
		np.random.shuffle(rho_s)
		print("rho:")
		print(rho_s)
		C_u_trans = np.dot(   np.dot(self.P , self.C_u )  ,self.P_T   )
		#C_u_trans = np.dot(self.P, np.dot(self.C_u,self.P_T))
		#print("C_u_trans")
		#print(C_u_trans)

		for i,ld in enumerate(rho_s):
			logpostList.append(self.getLogPostMultiple([rho_s[i],sigd_s[i],ld_s[i]]))

		n_logP = len(logpostList)
		logPmean = np.sum(np.array(logpostList)) / n_logP
		#print("logPmean:",logPmean)
		index_mean = (np.abs(logpostList - logPmean)).argmin()
		smallest = np.argmin(np.array(logpostList))
		print(smallest)
		print("rho,sig,l")
		print(np.exp(rho_s[smallest]),np.exp(sigd_s[smallest]),np.exp(ld_s[smallest]))
		rho_est = rho_s[smallest]
		sigd_est = sigd_s[smallest]
		ld_est = ld_s[smallest]

		plt.figure(figsize=(6, 4), dpi=100)
		#plt.plot(rho_s)
		#plt.contour([sigd_s,rho_s],logpostList)
		plt.scatter(np.exp(rho_s),logpostList,label="rho",alpha= 0.3)
		plt.scatter(np.exp(sigd_s),logpostList,label="sig",alpha= 0.3)
		plt.scatter(np.exp(ld_s),logpostList,label="l",alpha= 0.3)
		plt.legend()
		#plt.scatter(ld_s,logpostList,label="l")
		#plt.hist(logpostList,bins=140)
		#plt.yscale('log')
		plt.show()

		#result = scipy.optimize.minimize(fun=self.getLogPost,method='L-BFGS-B',bounds=((0.4,None),(1e-16,None),(1e-16,None)),x0=np.array([rho_est,sigd_est,ld_est]))
		#print("old result:")
		#print([np.exp(rho_est),np.exp(sigd_est),np.exp(ld_est)])
		#print("optimized result:")
		#print(result.x)

		samples = self.doMCMC([rho_est,sigd_est,ld_est])
		print('MCMC: ',samples)
		#return result.x

		return rho_est,sigd_est,ld_est
		#return rho_est,sigd_est,0.2



	def computePosterior(self,y_points,y_values):
		#y_points = [0.4,0.933333]
		C_e = self.get_C_e(len(y_points))
		P = self.getP(y_points)
		self.C_e = C_e
		pars = self.estimateHyperpar(y_points,y_values)
		rho=pars[0]
		sigd = pars[1]
		ld = pars[2]
		#rho,sigd,ld = self.estimateHyperpar(y_points,y_values)
		#rho = 1.1
		y_points = np.transpose(np.atleast_2d(np.array(y_points)))
		y_values = np.array(y_values)
		y = y_values - self.Pu
		#print("y points")
		#print(y_points)
		C_u = self.get_C_u()
		C_d = self.get_C_d(y_points,ld=ld,sigd=sigd)
		#C_d = 0
		P_T = np.transpose(P)
		rho = np.exp(rho)
		C_u_y =   C_u -   np.dot(  np.dot(C_u, P_T)   ,   np.dot( np.linalg.inv(  1/(rho*rho)  * (C_d + C_e)   +  np.dot(P, np.dot(C_u, P_T)   ) ) , np.dot(P,C_u))   )
		self.C_u_yDiag = np.sqrt(np.diagonal(C_u_y))
		C_u_inv = np.linalg.inv(C_u)
		# t0 = time.time()
		CdplusCeInv = np.linalg.inv(C_d+C_e)
		# t1 = time.time()
		# print("numpy inv:")
		# print(CdplusCeInv)
		# print(t1-t0)
		# t0 = time.time()
		CdplusCe_L = np.linalg.cholesky(C_d+C_e)
		CdplusCe_L_T = np.transpose(CdplusCe_L)
		#CdplusCeInv = np.dot(np.linalg.inv(CdplusCe_L_T),np.linalg.inv(CdplusCe_L))
		# t1 = time.time()
		# print("cholesky inv:")
		# print(CdplusCeInv)
		# print(t1-t0)
		#C_u_y = np.linalg.inv(    rho*rho * np.dot(P_T ,  np.dot(CdplusCeInv, P)) + C_u_inv )
		#print(C_u_y)
		u_mean = self.get_U_mean().get_local()
		#C_u_y = C_u   -   np.dot(C_u, P_T) * np.linalg.inv(  1/(rho*rho)  * (C_d + C_e)   +  np.dot(P, np.dot(C_u, P_T)   ) )  * np.dot(P,C_u)
		#u_mean_y = np.dot(C_u_y,  np.dot( rho*P_T   ,  np.dot( np.linalg.inv(C_d+C_e) , y_values))  +   np.dot(np.linalg.inv(C_u), u_mean ) )
		u_mean_y = rho * np.dot(C_u_y , (  rho * np.dot(np.dot(P_T  , CdplusCeInv)  , y_values)  + np.dot(C_u_inv , u_mean)  ))
		print(u_mean)

		#u_mean_y = u_mean + np.dot(np.dot(C_u,P_T) , np.linalg.inv(  1/(rho*rho)  * (C_d + C_e)   +  np.dot(P, np.dot(C_u, P_T)   ) ) )

		u_mean_y = u_mean +np.dot(       np.dot(    np.dot(C_u,P_T) , np.linalg.inv(  1/(rho*rho)  * (C_d + C_e)   +  np.dot(P, np.dot(C_u, P_T)   ) )  )     , y)
		#u_mean_y = rho * u_mean_y

		posteriorGP = np.random.multivariate_normal(
			mean = u_mean_y, cov=C_u_y,
			size=self.nMC)
		return u_mean_y,posteriorGP




	def computePosteriorMultipleY(self,y_points,y_values):
		""" here, y_values is a vector of different measurement sets. """

		C_e = self.get_C_e(len(y_points))
		P = self.getP(y_points)
		self.C_e = C_e
		pars = self.estimateHyperpar(y_points,self.yVectors)
		rho=pars[0]
		sigd = pars[1]
		ld = pars[2]

		y_points = np.transpose(np.atleast_2d(np.array(y_points)))
		y_values = np.array(y_values)

		#print("summed values:")
		sum_y = np.sum(y_values,axis=0)
		print("ysum:_")
		print(sum_y)
		#print(np.shape(self.y_values))
		print(np.shape(sum_y))

		C_u = self.get_C_u()
		C_u_inv = np.linalg.inv(C_u)
		C_d = self.get_C_d(y_points,ld=ld,sigd=sigd)

		P_T = np.transpose(P)
		rho = np.exp(rho)
		CdplusCeInv = np.linalg.inv(C_d+C_e)
		C_u_y = np.linalg.inv(    rho*rho * self.no * np.dot(P_T ,  np.dot(CdplusCeInv, P)) + C_u_inv )
		self.C_u_yDiag = np.sqrt(np.diagonal(C_u_y))




		CdplusCe_L = np.linalg.cholesky(C_d+C_e)
		CdplusCe_L_T = np.transpose(CdplusCe_L)

		u_mean = self.get_U_mean()
		u_mean_y = rho * np.dot(C_u_y , (  rho * np.dot(np.dot(P_T  , CdplusCeInv)  , np.transpose(sum_y))  + np.dot(C_u_inv , u_mean)  ))


		posteriorGP = np.random.multivariate_normal(
			mean = u_mean_y, cov=C_u_y,
			size=self.nMC)
		return u_mean_y,posteriorGP





	def getMCErrors(self):
		nMCList = np.logspace(1,5,200,dtype=int)
		#print('nMCLISTA')
		#print(nMCList)
		errorNormList = []
		errorNormListVar = []
		for nMC in nMCList:
			#print(nMC)
			self.nMC = nMC
			U,A = self.doFEM()
			U_mean = self.get_U_mean()
			U_mean_verbose = self.get_U_mean_verbose()

			#C_u = solver.get_C_u()
			priorSamples = self.samplePrior()
			#print('priorSamples')
			#print(np.transpose(priorSamples))


			sigL =[]
			muL = []
			DeltaL = []
			for i in range(self.ne+1):
				mu,sig,Delta = self.doMC(np.transpose(priorSamples)[i])
				sigL.append(sig)
				muL.append(mu)
				DeltaL.append(Delta)


			error_mean = U_mean - np.array(muL)
			error_var = np.square(np.array(self.C_u_yDiag)) - np.square(np.array(sigL))

			error_norm_var = np.sqrt(np.sum(np.square(error_var)))
			error_norm = np.sqrt(np.sum(np.square(error_mean)))
			errorNormList.append(error_norm)
			errorNormListVar.append(error_norm_var)
			print(nMC)
			#print(error_norm)

		print(np.array(self.C_u_yDiag))
		print(np.array(sigL))
		def myExpFunc(x, a, b):
			return a * np.power(x, b)
		popt, pcov = scipy.optimize.curve_fit(myExpFunc, nMCList, errorNormList)
		f = plt.figure(figsize=(6, 4), dpi=100)
		plt.plot(nMCList, errorNormList, nMCList, myExpFunc(nMCList, *popt), linestyle='-', color = 'black',lw = 1.0,label='mean error convergence.')
		#plt.plot(nMCList, errorNormListVar, linestyle='-', color = 'black',lw = 1.0,label='mean error convergence.')
		plt.yscale('log')
		plt.xscale('log')
		plt.grid()
		plt.rc('text', usetex=True)
		plt.rc('font',family='sans-serif')
		plt.ylabel(r'$\lVert \bar{u}-u_{MC} \rVert$',fontsize=16)
		plt.xlabel(r'number of MC points',fontsize=16)
		f.savefig("MCerrorConv.pdf", bbox_inches='tight')
		plt.show()
		return errorNormList



solver = solverClass(nMC=10)
U,A = solver.doFEM()
U_mean = solver.get_U_mean()
priorSamples = solver.samplePrior()
sigL =[]
muL = []
DeltaL = []
for i in range(solver.ne+1):
	mu,sig,Delta = solver.doMC(np.transpose(priorSamples)[i])
	sigL.append(sig)
	muL.append(mu)
	DeltaL.append(Delta)
error_mean = U_mean - np.array(muL)



#print(priorSamples[0])
#print(len(priorSamples[0]))
n_obs = 5+2
idx = np.round(np.linspace(0, len(priorSamples[0])-1, n_obs)).astype(int)
y_values_prior = [priorSamples[0][i] for i in idx]
y_values=[0.02393523,0.04423292, 0.06159137, 0.08335314, 0.09902092, 0.11984335,
 0.13244231, 0.14318289, 0.15658694, 0.17112178, 0.1784107,  0.1832616,
 0.19096354, 0.19309523, 0.19291609, 0.20442705, 0.20447705, 0.20763928,
 0.20639289, 0.20496173, 0.19824301, 0.19479635, 0.19457483, 0.19123965,
 0.18479936, 0.17647887, 0.16504261, 0.1598612 , 0.1507061,  0.14170226,
 0.12740095, 0.11301528, 0.09990149 ,0.08139896, 0.06955964, 0.05383899,
 0.03363033, 0.01623922]

#y_values = np.array(y_values)# + noise
y_values = y_values_prior[1:-1]
#noise = np.random.normal(0,1e-2,len(y_values))
#y_values = y_values + noise

print('values:')
print(y_values)
y_points = [solver.coordinates.tolist()[i] for i in idx][1:-1]


#y_values = [0.125,0.26,0.28,0.31,0.30,0.31,0.28,0.225,0.125]
#y_points = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#y_values = [x+0.3 for x in y_values]
#y_points = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#y_values = [0.12,0.26,0.28,0.34,0.30,0.34,0.24,0.225,0.15]
#y_points = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#y_values = [x-0.1 for x in y_values]

#y_values = [0.12,0.28,0.30,0.24,0.15]
#y_points = [0.1,0.3,0.5,0.7,0.9]
#y_values = [x-0.1 for x in y_values]

#y_values = y_values[5:]
#y_points = y_points[5:]

#y_values = solver.create_fake_data(y_points,y_values)
###### multiple observations
y_values_list = []
solver.no = 2
for i in range(solver.no):
	noise = np.random.normal(0,2.5e-3,len(y_values))
	y_values_list.append(y_values + noise)
solver.yVectors = y_values_list

############################


solver.y_points = y_points
solver.y_values = y_values
#solver.y_values = solver.create_fake_data(solver.y_points)
#u_mean_y,posterior_samples = solver.computePosterior(solver.y_points, solver.y_values)
u_mean_y,posterior_samples = solver.computePosteriorMultipleY(solver.y_points, y_values_list)
print("u_mean_y:")
print(u_mean_y)
error_var = np.square(np.array(solver.C_u_yDiag)) - np.square(np.array(sigL))









f = plt.figure(figsize=(6, 4), dpi=100)
plt.plot(solver.coordinates, np.transpose(U_mean), linestyle='-', color = 'black',lw = 1.0,label='FEM mean')
plt.fill_between(np.transpose(solver.coordinates)[0], np.array(muL)+1.96*solver.C_uDiag, np.array(muL)-1.96*solver.C_uDiag,color = 'blue',alpha=0.3,label='$2\sigma$ FEM')

#plt.plot(solver.coordinates, np.transpose(U_mean_verbose), linestyle='-.', color = 'red',lw = 1.0)
#plt.plot(solver.coordinates, np.transpose(priorSamples[10:390]), linestyle='-',lw = 0.4,color='black', alpha=0.35)
#plt.plot(np.transpose(solver.coordinates)[0], np.array(muL), linestyle='-.',color = 'black',lw = 3.0, label='Mean MC')
#plt.plot(np.transpose(solver.coordinates)[0], error_var, linestyle='-.',color = 'green',lw = 2.0, label='Mean error')
#plt.plot(solver.coordinates, np.transpose(posterior_samples[10:150]), linestyle='-',lw = 0.2,color='black', alpha=0.4)
#plt.plot(np.transpose(solver.coordinates)[0], np.transpose(u_mean_y)-1.96*solver.C_u_yDiag, linestyle='-.',color = 'green',lw = 1.0,label='2sig')
#plt.plot(np.transpose(solver.coordinates)[0], np.transpose(u_mean_y)+1.96*solver.C_u_yDiag, linestyle='-.',color = 'green',lw = 1.0)
plt.fill_between(np.transpose(solver.coordinates)[0], np.transpose(u_mean_y)+1.96*solver.C_u_yDiag, np.transpose(u_mean_y)-1.96*solver.C_u_yDiag,color = 'green',alpha=0.3,label='$2\sigma$ Posterior')


plt.plot(solver.coordinates, np.transpose(u_mean_y), linestyle='-', color = 'green',lw = 2.0,label='Posterior mean')
#plt.scatter(solver.y_points, solver.y_values,label='observations')
for obs in y_values_list:
	plt.scatter(solver.y_points, obs,s=2.5, color = 'black',alpha=0.4)

#plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)-1.96*np.array(sigL), linestyle='-',color = 'red',lw = 1.0,label='2sig MC')
#plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)+1.96*np.array(sigL), linestyle='-',color = 'red',lw = 1.0)

#plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)-1.96*solver.C_uDiag, linestyle='-.',color = 'black',lw = 2.0,label='$2\sigma$ confidence band')
#plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)+1.96*solver.C_uDiag, linestyle='-.',color = 'black',lw = 2.0)

plt.legend()
plt.xlim([solver.dom_a, solver.dom_b])
plt.grid()
#plt.show()
f.savefig("Result.pdf", bbox_inches='tight')
#solver.estimateHyperpar(y_points, y_values)
solver.samplef()

#plt.pause(0.01) # Pause for interval seconds.
#input("hit[enter] to end.")
#plt.close('all') # all open plots are correctly closed after each run
