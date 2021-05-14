import sys
from fenics import *
import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
import time
import seaborn as sns
import numpy as np
#np.seterr(all='ignore')
import scipy
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
#
from numba import jit


class solverClass:
	def __init__(self, nMC = 200):
		self.dom_a = 0.0 #domain boundaries
		self.dom_b = 1.0

		self.ne	= 120 #number of elements
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



	def draw_FEM_samples(self, x_vals):
		#print("fem mean f")
		#print(self.mean)
		cov = self.exponentiated_quadratic(x_vals, x_vals, lf=self.lf, sigf=self.sigf)  # Kernel of data points
		discrete_f = np.random.multivariate_normal(
			mean = self.mean , cov=cov,
			size=1)
		return discrete_f



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
		#self.integratedTestF = np.array(assemble(Constant(self.mean[1]) * self.v * dx).get_local()) #* self.ne
		self.integratedTestF = np.array(assemble(Constant(1.0) * self.v * dx).get_local()) #* self.ne

		#print(self.integratedTestF)
		c_f = self.exponentiated_quadratic(self.coordinates,
			 self.coordinates, lf=self.lf, sigf=self.sigf)
		C_f = np.zeros((self.ne+1,self.ne+1))
		#print('coords:')
		#print(self.coordinates)
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

		#A_inv = np.linalg.inv(A)
		A_inv = np.linalg.solve(A,ident)
		thresh = 1e-16
		#ind_smaller_thresh = A_inv < thresh
		#A_inv[ind_smaller_thresh] = 0
		#C_u = A_inv * C_f * np.transpose(A_inv)
		C_u = np.dot( np.dot(A_inv,C_f), np.transpose(A_inv))
		C_u = C_u + 1e-16*(self.sigf**2)*ident
		c_u = np.transpose(self.integratedTestF) * C_u * self.integratedTestF
		self.C_u = C_u
		self.C_uDiag = np.sqrt(np.diagonal(self.C_u))

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
		mu = np.sum(samples) / n
		var = (1/(n-1)) * np.sum( np.square(samples-mu) )
		sig = np.sqrt(var)
		d = -norm.ppf((1-gamma)/2)
		Delta = sig

		return (mu, sig, Delta)




	def doMCMC(self,w0):
		sig_q = 0.005
		cov = sig_q**2 * np.identity(len(w0))
		numSamp = 5000
		samples = []
		#w0 =[1,1,1]
		samples.append(w0)
		wi = w0

		for i in list(range(numSamp))[1:]:
			#proposal =  samples[i-1] + np.random.normal(loc=0.0, scale = sig_q, size=len(w0))
			proposal = np.random.multivariate_normal(
			mean = wi, cov=cov)
			proposal = abs(proposal)
			alpha = np.min((0, (self.getLogPost(np.log(proposal) - self.getLogPost(np.log(wi)))  )))
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
		plt.figure(figsize=(6, 4), dpi=100)
		plt.plot(np.transpose(samples)[0],label=['rho','sig',"l"])
		#plt.yscale('log')
		#plt.xscale('log')

		#plt.hist(np.transpose(samples),bins=30)
		plt.grid()
		plt.legend()


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
		#ld = 0.00103
		#sigd = 0.009340
		C_d = self.exponentiated_quadratic(y_points, y_points, lf=ld, sigf=sigd)
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

		#rho = np.exp(rho)  ##wieso?!

		C_u_trans = np.dot(   np.dot(self.P , self.C_u )  ,self.P_T   )
		#C_u_trans = np.dot(self.P, np.dot(self.C_u,self.P_T))

		K_y = rho * rho *  C_u_trans      + self.get_C_d(y_points = y_points,ld=ld,sigd=sigd)+ self.C_e
		#K_y = rho * rho *  C_u_trans


		#K_y_inv = np.linalg.inv(K_y)
		ny = len(y_values)
		#K_y_L = cho_factor(K_y)[0]
		#K_y_L_T = np.transpose(K_y_L)
		#K_y_inv = np.dot(np.linalg.inv(K_y_L_T),np.linalg.inv(K_y_L))
		#y = rho * self.Pu - y_values
		#K_y_det = np.linalg.det(K_y)
		#logpost = 0.5 * np.dot(np.transpose(y), np.dot(K_y_inv , y)  )  +0.5*np.log(K_y_det) + ny/2 * np.log(2* np.pi) #Version Paper
		#print(K_y @ np.dot(K_y_inv , y) - y)


		y = y_values - rho * self.Pu
		L = cho_factor(K_y)
		K_y_inv_y = cho_solve(L, y)
		Log_K_y_det = 2 * np.sum(np.log(np.diag(L[0])))
		logpost2 = 0.5 * (np.dot(np.transpose(y), K_y_inv_y )  +Log_K_y_det + ny * np.log(2* np.pi))#Version Paper
		print(K_y @ K_y_inv_y - y)




		#logpost = 0.5 * np.dot(np.transpose(y), np.dot(K_y_inv , y)  )  +0.5*np.log(K_y_det) + ny/2 * np.log(2* np.pi) #Version Paper

		#logpost = ny * np.pi + 0.5*K_y_det + 0.5* np.dot(np.transpose(y), np.dot(K_y_inv, y) ) #Version Römer
		return logpost2


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
		#rho = np.exp(rho)  ##wieso?!
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
		ld_s = np.random.uniform(0.0001,1,1000)
		sigd_s = np.random.uniform(0.0001,0.5,1000)
		#rho_s = np.random.uniform(-0.6,3,200000)
		#rho_s = np.linspace(-0.1,0.5,200000)
		rho_s = np.linspace(0.8,1.8,1000)
		rho_s = np.random.uniform(0.6,1.8,1000)
		#rho_s = np.exp(rho_s)  ##wieso?!
		#ld_s = np.log(ld_s)
		#sigd_s = np.log(sigd_s)

		C_u_trans = np.dot(   np.dot(self.P , self.C_u )  ,self.P_T   )
		#C_u_trans = np.dot(self.P, np.dot(self.C_u,self.P_T))
		#print("C_u_trans")
		#print(C_u_trans)

		for i,ld in enumerate(rho_s):
			logpostList.append(self.getLogPost([rho_s[i],sigd_s[i],ld_s[i]]))

		n_logP = len(logpostList)
		logPmean = np.sum(np.array(logpostList)) / n_logP
		#print("logPmean:",logPmean)
		index_mean = (np.abs(logpostList - logPmean)).argmin()
		smallest = np.argmin(np.array(logpostList))
		#print(smallest)
		#print("rho,sig,l")
		#print(rho_s[smallest],sigd_s[smallest],ld_s[smallest])
		rho_est = rho_s[smallest]
		sigd_est = sigd_s[smallest]
		ld_est = ld_s[smallest]
		plt.figure(figsize=(6, 4), dpi=100)

		#plt.contour([sigd_s,rho_s],logpostList)
		plt.scatter(rho_s,logpostList,label="rho")
		#plt.scatter(sigd_s,logpostList,label="sig")
		#plt.scatter(ld_s,logpostList,label="l")
		#plt.hist(logpostList,bins=100)
		#plt.yscale('log')
		plt.show()

		result = scipy.optimize.minimize(fun=self.getLogPost,bounds=((1e-16,None),(1e-16,None),(1e-16,None)),x0=np.array([1,1,1]))
		print("old result:")
		print([rho_est,sigd_est,ld_est])
		print("optimized result:")
		print(result)

		#self.getLogPostDeriv(rho_est,sigd_est,ld_est)
		#self.getLogPost(rho_est,sigd_est,ld_est)
		#return 1,1e-16,0.5
		return result.x
		#return -0.2,0,0
		#print('MCMC from here on')
		#print([rho_est,sigd_est,ld_est])
		#samples = self.doMCMC([rho_est,sigd_est,ld_est])
		#print('MCMC: ',samples)
		#return samples
		#return rho_est,sigd_est,ld_est

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
		#print("y points")
		#print(y_points)
		C_u = self.get_C_u()
		C_d = self.get_C_d(y_points,ld=ld,sigd=sigd)
		#C_d = 0
		P_T = np.transpose(P)

		C_u_y =   C_u -   np.dot(  np.dot(C_u, P_T)   ,   np.dot( np.linalg.inv(  1/(rho*rho)  * (C_d + C_e)   +  np.dot(P, np.dot(C_u, P_T)   ) ) , np.dot(P,C_u))   )
		self.C_u_yDiag = np.sqrt(np.diagonal(C_u_y))
		C_u_inv = np.linalg.inv(C_u)
		# t0 = time.time()
		# CdplusCeInv = np.linalg.inv(C_d+C_e)
		# t1 = time.time()
		# print("numpy inv:")
		# print(CdplusCeInv)
		# print(t1-t0)
		# t0 = time.time()
		CdplusCe_L = np.linalg.cholesky(C_d+C_e)
		CdplusCe_L_T = np.transpose(CdplusCe_L)
		CdplusCeInv = np.dot(np.linalg.inv(CdplusCe_L_T),np.linalg.inv(CdplusCe_L))
		# t1 = time.time()
		# print("cholesky inv:")
		# print(CdplusCeInv)
		# print(t1-t0)
		#C_u_y = np.linalg.inv(    rho*rho * np.dot(P_T ,  np.dot(CdplusCeInv, P)) + C_u_inv )
		#print(C_u_y)
		u_mean = self.get_U_mean()
		#C_u_y = C_u   -   np.dot(C_u, P_T) * np.linalg.inv(  1/(rho*rho)  * (C_d + C_e)   +  np.dot(P, np.dot(C_u, P_T)   ) )  * np.dot(P,C_u)
		u_mean_y = np.dot(C_u_y,  np.dot( rho*P_T   ,  np.dot( np.linalg.inv(C_d+C_e) , y_values))  +   np.dot(np.linalg.inv(C_u), u_mean ) )
		u_mean_y = np.dot(C_u_y , (  rho * np.dot(np.dot(P_T  , CdplusCeInv)  , y_values)  + np.dot(C_u_inv , u_mean)  ))


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



solver = solverClass(nMC=2000)
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
n_obs = 9
idx = np.round(np.linspace(0, len(priorSamples[0])-1, n_obs)).astype(int)
y_values_prior = [priorSamples[0][i] for i in idx]
y_points = np.linspace(0.1,0.9,6).tolist()
y_values=[0.0004920659883738306, 0.17529629477383774, 0.26500738352141456, 0.26730432282836786, 0.17717424922549324, -0.0004905481144322452]

y_values = np.array(y_values)# + noise
y_values = y_values_prior[1:-1]
noise = np.random.normal(0,5e-3,len(y_values))
y_values = y_values + noise
print('values:')
print(y_values)
y_points = [solver.coordinates.tolist()[i] for i in idx][1:-1]
#print("points:")
#print(y_points)
#print('indexes: ',idx)
solver.y_points = y_points
solver.y_values = y_values
u_mean_y,posterior_samples = solver.computePosterior(y_points, y_values)
#errorNormList = solver.getMCErrors()
error_var = np.square(np.array(solver.C_u_yDiag)) - np.square(np.array(sigL))


plt.figure(figsize=(6, 4), dpi=100)
plt.plot(solver.coordinates, np.transpose(U_mean), linestyle='-', color = 'black',lw = 1.0,label='Mean analyt.')
#plt.plot(solver.coordinates, np.transpose(U_mean_verbose), linestyle='-.', color = 'red',lw = 1.0)
#plt.plot(solver.coordinates, np.transpose(priorSamples[10:390]), linestyle='-',lw = 0.4,color='black', alpha=0.35)
#plt.plot(np.transpose(solver.coordinates)[0], np.array(muL), linestyle='-.',color = 'black',lw = 3.0, label='Mean MC')
plt.plot(np.transpose(solver.coordinates)[0], error_var, linestyle='-.',color = 'green',lw = 2.0, label='Mean error')
plt.plot(solver.coordinates, np.transpose(posterior_samples[10:150]), linestyle='-',lw = 0.2,color='black', alpha=0.4)
plt.plot(np.transpose(solver.coordinates)[0], np.transpose(u_mean_y)-2*solver.C_u_yDiag, linestyle='-.',color = 'black',lw = 2.0,label='2sig')
plt.plot(np.transpose(solver.coordinates)[0], np.transpose(u_mean_y)+2*solver.C_u_yDiag, linestyle='-.',color = 'black',lw = 2.0)

plt.plot(solver.coordinates, np.transpose(u_mean_y), linestyle='-', color = 'red',lw = 2.0,label='Posterior mean')
plt.scatter(y_points, y_values,label='observations')

plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)-2*np.array(sigL), linestyle='-',color = 'red',lw = 3.0,label='2sig MC')
plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)+2*np.array(sigL), linestyle='-',color = 'red',lw = 3.0)

plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)-2*solver.C_uDiag, linestyle='-.',color = 'black',lw = 2.0,label='2sig')
plt.plot(np.transpose(solver.coordinates)[0], np.array(muL)+2*solver.C_uDiag, linestyle='-.',color = 'black',lw = 2.0)

#plt.fill_between((solver.coordinates)[0], np.array(muL)-np.array(DeltaL), np.array(muL)+np.array(DeltaL),color = 'blue',label='2sig')
plt.legend()
plt.xlim([solver.dom_a, solver.dom_b])
plt.grid()
plt.show()
#solver.estimateHyperpar(y_points, y_values)


#plt.pause(0.01) # Pause for interval seconds.
#input("hit[enter] to end.")
#plt.close('all') # all open plots are correctly closed after each run
