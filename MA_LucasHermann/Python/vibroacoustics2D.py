import sys
from fenics import *

import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import shift # shift and align are for plotting multiple axes in MPL while having the same zero.
import align
import numpy as np
import scipy
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve #for cholesky decomp

import scitools as st

import scitools.BoxField

import pickle
PIK = "factor.dat" # solutions will be pickled and saved to that filename. Free to choose right here!
#

class Params:
    # parameter class is introduced for Fenics Expression classes which couldn't 
    # access the parameters in their surrounding class otherwise.
	dom_a = 0.0 #domain boundaries
	dom_b = 0.85

	factor = 1 # factor for sine wave: 2sin(factor * pi *x)


class solverClass:
	def __init__(self):
		self.dom_a =Params.dom_a# 0.0  #domain boundaries, fethcing from params class
		self.dom_b = Params.dom_b#0.85

		self.SourceBoundary = 1 #determines on which boundary part the source is applied. 0 left, 1 right

		self.a,self.b = 24,24
		self.mesh = UnitSquareMesh(self.a,self.b) #define mesh
		self.V = FunctionSpace(self.mesh, "Lagrange", 1) # Function space

		#parameters for acoustic medium, air:
		self.rho = 1.2
		self.f =320
		c = 340
		self.omega = 2* np.pi * self.f
		self.k = self.omega / c # wave number
		self.g = self.rho * self.omega**2 # part of the boundary term for the variational problem. Displacement u is introduced below

		factor = Params.factor # fetching from Params class
		class MyExpression0(UserExpression):
			## old class. remove asap
			def eval(self, value, x):
				value[0] = 0.00001 *np.sin(factor*x[1]) #with sine
				#value[0] = 1 #without sine
			def value_shape(self):
				return ()	

		class MyExpression1(UserExpression):
			## Fenics expression to model the piston displacement BC function
			def eval(self, value, x):
				val = 20 *np.sin(factor*np.pi*x[1])
				val = 0.00001 *np.sin(2*np.pi*x[1]) + 0.000005 *np.sin(4*np.pi*x[1])
				val = 0.00001 *np.sin(Params.factor*np.pi*x[1]) + 0.000005 *np.sin(2*Params.factor*np.pi*x[1])
				if x[0] >(1.0-1e-8):
					value[0] = val
					#value[0] = 0.00001 #overwrite sine using a constant function
				else:
					value[0] = 0.0 #only the boundary is non-zero. evry other point is zero.
			def value_shape(self):
				return ()

		self.f0 = MyExpression1()
		self.f1 = MyExpression1() # f0 and f1 are both used. needs to be changed to a single one since they're the same
		
		# arrays to store solution data:
		self.fArray = []
		self.f_bar_list= []
		self.solArray = []

		phi = np.zeros((self.a+1,self.b+1))
		phi[:,-1] = 1
		phi_vector = phi.reshape((self.a+1)*(self.b+1))
		psi = Function(self.V)
		psi.vector()[:] = phi_vector[vertex_to_dof_map(self.V)]
		print(psi.vector().get_local())
		self.fmeanGP = psi

		
		self.UMeanOutput=[]
		self.UCutOutput = []
		self.UCutVarOutput=[]
		self.PriorOutput = []
		self.PriorVarOutput=[]


	def exponentiated_quadratic_log(self, xa, xb, l, sig):
		""" Sq. Exp. Kernel: expects input parameters as log(par). """
		return np.exp(2*sig) * np.exp(-0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * np.exp(-2*l))


	def matern_log(self, xa, xb, l, sig):
		""" Matern Kernel: expects input parameters as log(par). """
		r = scipy.spatial.distance.cdist(xa, xb, 'euclidean')
		#return np.exp(2*sig) * np.exp(-1 * scipy.spatial.distance.cdist(xa, xb, 'euclidean') * np.exp(-1*l)) #nu = 1/2
		return np.exp(2*sig) * ((1 + np.sqrt(3) * r *np.exp(-1*l)) *  np.exp(-1 * np.sqrt(3) * r * np.exp(-1*l))   ) # nu = 3/2
		#return np.exp(2*sig) * ((1 + np.sqrt(5) * r *np.exp(-1*l) +  5*r*r/3*np.exp(-1*l)*np.exp(-1*l)  ) *  np.exp(-1 * np.sqrt(5) * r * np.exp(-1*l))   ) # nu = 5/2


	def get_U_mean(self):
		"""
		calculates the mean for the FEM solution prior GP
		"""

		sb = self.SourceBoundary
		u_mean = Function(self.V)
		U_mean = u_mean.vector()
		#print(self.mean)
		#print(self.mean[1])


		f0 = self.f0
		#self.d_mean = 0.0005# * f0
		g = self.rho * self.omega**2 #* self.d_mean
		b_mean = assemble(self.v*g*f0*self.ds(sb)) # because for the mean, f = 1. (1 * v * dx)
		f_bar_plot = assemble(self.v*f0*self.ds(sb))

		# print("f_bar: links")
		# f_bar = [x for x in b_mean.get_local() if x!=0.0]
		# self.f_bar = f_bar
		# print(f_bar)
		# print(len(f_bar))
		# plt.figure
		# plt.plot(f_bar)
		# plt.title("$f_bar$")
		# plt.show()

		# self.f_bar_list.append(f_bar)
		self.b_mean = b_mean
		#self.bc.apply(b_mean)
		A = self.A
		solve(A,U_mean,b_mean)
		self.U_mean = U_mean

		X = 0; Y = 1; Z = 0
		u_box = st.BoxField.dolfin_function2BoxField(u_mean,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (0.0,0.7)
		x,umeanval,y_fixed,snapped = u_box.gridline(start, direction = X)
		self.U_mean_cut = umeanval


		#u = Function(self.V)
		#u.vector().set_local(U_mean.tolist()[0])

		#c = plot(u_mean,title="Prior mean")
		#plt.title("prior mean")
		#plt.colorbar(c)
		#plt.plot_surface(x_list,y_list,np.transpose(discrete_d))
		#plt.show()
		return U_mean



	def get_C_u(self):
		"""
		infer FEM prior covariance
		"""

		C_f = self.get_C_f()
		
		A = self.A
		#ident = np.identity(len(self.coordinates))
		ident = np.identity(len(self.V.tabulate_dof_coordinates()))
		A = A.array()

		A_inv = np.linalg.inv(A)  ##needs to be choleskied!
		#A_inv = np.linalg.solve(A,ident)
		C_u = np.dot( np.dot(A_inv,C_f), np.transpose(A_inv))
		self.C_uDiag = np.sqrt(np.diagonal(C_u))
		condBefore = np.linalg.cond(C_u)
		C_u = C_u + 1e-8*ident
		condAfter = np.linalg.cond(C_u)
		c_u = np.transpose(self.integratedTestF) * C_u * self.integratedTestF
		self.C_u = C_u
		print("Cond:")
		print(condBefore,condAfter)
		self.C_uDiag = np.sqrt(np.diagonal(self.C_u))

		#C_f = C_f + 1e-10*ident
		#print("C_u:")
		#print(C_u)
		#########################################

		discrete_u = np.random.multivariate_normal(
			mean = self.U_mean , cov=C_u,
			size=1)
		##mean = 0.01*np.zeros(len(self.V.tabulate_dof_coordinates()))
		u = Function(self.V)
		u.vector().set_local(discrete_u.tolist()[0])

		u_sig = Function(self.V)
		u_sig.vector().set_local(self.C_uDiag.tolist())


		X = 0; Y = 1; Z = 0
		u_box = st.BoxField.dolfin_function2BoxField(u,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (0.0,0.7)
		x,uval,y_fixed,snapped = u_box.gridline(start, direction = X)
		self.x_cut = x

		#cut for diagonal:
		sig_cut = Function(self.V)
		sig_cut.vector()[:] = self.C_uDiag[vertex_to_dof_map(self.V)]
		X = 0; Y = 1; Z = 0
		u_sig_box = st.BoxField.dolfin_function2BoxField(u_sig,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (0.0,0.7)
		x,uvalsig,y_fixed,snapped = u_sig_box.gridline(start, direction = X)
		self.sig_cut = uvalsig
		#plt.figure()
		#plt.plot(x,uval)
		#plt.title("Prior sample cut")
		#plt.show()

		self.solArray.append(uval)

		# plt.figure()
		# plt.plot(x,uval)
		# plt.title("Prior sample cut")
		fig = plt.figure()
		c=plot(u_sig)
		plt.colorbar(c)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		
		fig.savefig("VarField.pdf", bbox_inches='tight')
		plt.close(fig)
		# plt.figure()
		# plt.plot(x,uvalsig)
		# plt.title("sig Prior sample cut")

		# plt.show()


		print("Cu Diag shape:")	
		print(np.shape(self.C_uDiag))
		CuDiag = Function(self.V)
		CuDiag.vector().set_local(self.C_uDiag.tolist())
		X = 0; Y = 1; Z = 0
		Cu_box = st.BoxField.dolfin_function2BoxField(CuDiag,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (0.0,0.7)
		x,CuDiagval,y_fixed,snapped = u_box.gridline(start, direction = X)
		self.C_uDiag = CuDiagval
######################################
		return C_u
		#return c_u


	def get_C_f(self):
		"""
		compute forcing covariance
		"""
		
		ds = self.ds
		sb = self.SourceBoundary
		self.integratedTestF = np.array(assemble(Constant(1.0) * self.v * ds(sb)).get_local()) #* self.ne
		
		print("intTestF:")
		print(self.integratedTestF)
		print(np.shape(self.integratedTestF))
		y_list = np.expand_dims(np.linspace(0, 1, self.b+1), 1)
		x_list = np.expand_dims(np.linspace(0, 1, self.a+1), 1)

		c_f = self.exponentiated_quadratic_log(self.dof_coordinates, self.dof_coordinates, l=np.log(0.3), sig=np.log(5))
		c_f = self.matern_log(self.dof_coordinates, self.dof_coordinates, l=np.log(0.8), sig=np.log(5))
		self.c_f = c_f 

		# assembling C_f in FEM:
		C_f = np.zeros(((self.a+1)*(self.b+1),(self.a+1)*(self.b+1)))
		for i in range((self.a+1)*(self.b+1)):
			for j in range((self.a+1)*(self.b+1)):
				C_f[i,j] = self.integratedTestF[i] * c_f[i,j] * self.integratedTestF[j]


		#plt.figure()
		#plot(self.mesh)
		#plt.title("mesh")
		#plt.show()
		ident = np.identity(len(self.V.tabulate_dof_coordinates()))
		#mean = [0.0001 *np.sin(self.factor*y) for y in np.ones(len(self.V.tabulate_dof_coordinates())).tolist()]
	
		#np.sin*np.ones(len(self.V.tabulate_dof_coordinates()))
	#	discrete_d = np.random.multivariate_normal(
		#	mean = np.zeros(len(self.V.tabulate_dof_coordinates())) , cov=C_f,
	#		size=1)

		#u = Function(self.V)

		#u.vector().set_local(discrete_d.tolist()[0])
		#print(discrete_d.tolist())

		#X = 0; Y = 1; Z = 0
		#u_box = st.BoxField.dolfin_function2BoxField(u,self.mesh,(self.a,self.b),uniform_mesh=True)
		#start = (1.0,0.0)
		#start = (0.0,0.0)
		#x,uval,y_fixed,snapped = u_box.gridline(start, direction = Y)

		#self.fArray.append(uval)
		#self.x_f = x
	
		#plt.figure()
		#plt.plot(x,uval)
		#plt.title("forcing")
		#plt.show()

		######Different approach with Cholesky decomposition
		## Get cholesky decomposition (square root) of the
		## covariance matrix
		#L = np.linalg.cholesky(c_f+1e-7)
		## Sample 3 sets of standard normals for our test points,
		## multiply them by the square root of the covariance matrix
		sigma = 1
		#f_prior = np.dot(L, np.random.normal(size=1,scale=sigma))
		#
		### add a mean function
		#f_priorT = np.transpose(f_prior)
		#for i in range(number_of_functions):
		#    f_priorT[i]+=mean
		######################################
		#plt.figure()
		#c = plot(u)
		#plt.colorbar(c)
		#plt.title("forcing complete")
		#plt.show()
		self.C_f = C_f
		return C_f


	def samplef(self,n):
		"""
		sampling from the source GP for illustration purposes
		"""
		flist=[]
		f_mean = self.b_mean.get_local()
		#f_mean=self.fmeanGP.vector().get_local()
		f_mean_fenicsObj = interpolate(self.f1,self.V)
		f_mean = f_mean_fenicsObj.vector().get_local()*self.g
		#print("f_mean:")
		#print(f_mean)
		C_fDiag = np.sqrt(np.diagonal(self.C_f))
		fGP = np.random.multivariate_normal(
			mean = f_mean, cov=self.c_f,
			size=n)

		for i in range(n):
			f = Function(self.V)
			f.vector().set_local(fGP[i].tolist())
			flist.append(f)

		# fig1 = plt.figure(figsize=(6, 4), dpi=100)
		# #print("len flist:")
		# #print(len(flist))
		# #print(flist)
		# plot(flist[0])
		# plt.ylabel("$f(x)$")
		# plt.xlabel("$x$")
		# plt.grid()
		# #plt.legend()
		# #fig1.savefig("fGP.pdf", bbox_inches='tight')
		# #f.savefig("sqex_f_sampled.pdf", bbox_inches='tight')
		# plt.title("fGP")
		# plt.show()




		fig2 = plt.figure()
		fvalList = []
		for f in flist:
			X = 0; Y = 1; Z = 0
			f_box = st.BoxField.dolfin_function2BoxField(f,self.mesh,(self.a,self.b),uniform_mesh=True)
			start = (1.0,0.0)
			x,fval,y_fixed,snapped = f_box.gridline(start, direction = Y)
			fvalList.append(fval)
			#plt.plot(x,fval)
		
		X = 0; Y = 1; Z = 0
		f_box_mean = st.BoxField.dolfin_function2BoxField(f_mean_fenicsObj,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (1.0,0.0)
		x,fvalMean,y_fixed,snapped = f_box_mean.gridline(start, direction = Y)
		# self.fArray.append(uval)
		# self.x_f = x
		print("fvalMean:")
		print(fvalMean)
		fvalListTrans = np.transpose(fvalList)
		print(np.shape(fvalListTrans[0]))
		sigL = []
		muL = []
		DeltaL = []
		self.SourceMean = muL
		for point in fvalListTrans:
			mu, sig, Delta = self.doMC(point)
			sigL.append(sig)
			muL.append(mu)
			DeltaL.append(Delta)#
		print(muL)
		self.sourceVariance=sigL
		for sample in fvalList[0:10]:
			plt.plot(x,sample)
		#plt.plot(x, muL, color = 'black',lw = 2.0,label='2sig')
		plt.plot(x, fvalMean, color = 'black',lw = 2.0,label='2sig')

		plt.plot(x, fvalMean-1.96*np.array(sigL), linestyle='-.',color = 'black',lw = 2.0,label='2sig')
		plt.plot(x, fvalMean+1.96*np.array(sigL), linestyle='-.',color = 'black',lw = 2.0,label='2sig')
		plt.title("fGP cut")
		fig2.savefig("fGPCut.pdf", bbox_inches='tight')
		#plt.show()


	def doMC(self,samples):
		"""
		getting the moments from the distributions in a MC way
		"""
		gamma = 0.95
		n = len(samples)
		mu = np.sum(samples) / n
		var = (1/(n-1)) * np.sum( np.square(samples-mu) )
		sig = np.sqrt(var)
		d = -norm.ppf((1-gamma)/2)
		Delta = sig

		return (mu, sig, Delta)



	def getDispGP(self):
		"""
		not needed anymore
		"""
		y_list = np.expand_dims(np.linspace(0, 1, self.b), 1) # Vector of poitns in the mesh
		d_mean = 0.01*np.ones(len(y_list))
		cov = self.exponentiated_quadratic_log(y_list, y_list, l=np.log(0.15), sig=np.log(0.001))
		cov = self.matern_log(y_list, y_list, l=np.log(0.15), sig=np.log(0.001))
		discrete_d = np.random.multivariate_normal(
			mean = d_mean , cov=cov,
			size=1)
		#self.d_mean = d_mean
		#plt.figure()
		#plt.plot(y_list,np.transpose(discrete_d))
		#plt.show()
		return discrete_d



	def doFEM(self):
		"""
		basic FEM solver for the Helmholtz equation. Gives the mean solution for the prior
		"""
		# Define variational problem

		self.u = TrialFunction(self.V)
		self.v = TestFunction(self.V)

		#boundary_markers = FacetFunction('size_t', self.mesh)
		#factor = 2
		# class MyExpression0(UserExpression):
		# 	def eval(self, value, x):
		# 		#value[0] = np.abs(np.sin(factor*x[1]))
		# 		value[0] = 0.0001*np.sin(factor*x[1])
		# 	def value_shape(self):
		# 		return ()
		#x=np.linspace(0,1,100)
		f0 = self.f0
		#f0=1
		#plt.plot(x,np.abs(np.sin(factor*x)))
		#print("meshDim:")
		#print(self.mesh.topology().dim())
		boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1	, 0)

		tol = 1e-14
		class BoundaryX_L(SubDomain):
			self.tol = 1E-14
			def inside(self, x, on_boundary):
				return on_boundary and near(x[0], 0, tol)# and (x[1] < 0.3)

		class BoundaryX_R(SubDomain):
			def inside(self, x, on_boundary):
				return on_boundary and near(x[0], 1, tol)# and (x[1] < 0.3)

		def boundaryDiri(x):
			return x[0] > (1.0 - DOLFIN_EPS) and x[1] < 0.0 + DOLFIN_EPS

		self.bcDir = DirichletBC(self.V, Constant(0.0), boundaryDiri)

		boundary_markers.set_all(9999) # all markers default to zero.
		bxL = BoundaryX_L()
		bxR = BoundaryX_R()
		bxL.mark(boundary_markers,0) # boundary bxL is marked as "1"
		bxR.mark(boundary_markers,1)
		#g = rho * omega**2 * u
		#self.SourceBoundary = 1
		sb = self.SourceBoundary
		ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers) #ds (contrary to dS) describes only the boundaries. so: using ds(0) would mean using all boundaries. if we set markers, we can pick single boudnaries out of the pool.
		self.ds = ds
		#a = dot(grad(self.u), grad(self.v))*dx - self.k**2 * dot(self.u,self.v)*dx #variational Problem
		a = inner(nabla_grad(self.u), nabla_grad(self.v))*dx - self.k**2 * inner(self.u,self.v)*dx #variational Problem
		#L = self.v*self.g*ds(0) # L is nonzero only on theboudnary marked with "1". if ds(0) would be used, all two boundaries would be nonzero.
		L = (self.v*self.g*f0)*ds(sb)
		#L = self.v*self.g*ds(sb)
		A = assemble(a)
		b = assemble(L)
		#self.bcDir.apply(A, b)
		self.u = Function(self.V)
		U = self.u.vector()
		#solve(A, U, b)
		solve(A,U,b)
		#data = U.array()
		mean_index = int(self.a / 2)
		#mean_column = data[mean_index]for x in mesh.coordinates():


		self.dof_coordinates = self.V.tabulate_dof_coordinates()
		n = self.V.dim()
		gdim = self.mesh.geometry().dim()
		dofmap = self.V.dofmap()
		dofs = dofmap.dofs()
		print("dofCords:")
		print(self.dof_coordinates)
		
		# dof_x = self.V.tabulate_dof_coordinates().reshape((-1, gdim))
		# x = dof_x[:,0]
		# indices = np.where(np.logical_and(x > 0.26, x < 0.34))[0]
		# xs = dof_x[indices]
		# vals = U[indices]
		# maxval = np.max(vals)
		# vals/=3*maxval


		#u = u.vector()
		#plt.plot(np.abs(U))
		#print("u:")
		#print(U.get_local())

		#c=plot(self.u)
		#plt.colorbar(c)
		#plt.title("basic FEM without GP")
		#plt.show()
		#plt.figure()
		#plt.plot(np.linspace(0,1,len(vals)),vals.tolist())
		#plt.show()


		self.A = A


		self.x0 = interpolate(Expression("x[0]",degree=1),self.V)
		self.x1 = interpolate(Expression("x[1]",degree=1),self.V)


	def get_C_e(self,size):
		sige_square = 2.5e-5
		C_e = sige_square * np.identity(size)
		#C_e = np.zeros((size,size))
		self.C_e = C_e
		#print('C_e:')
		#print(C_e)
		return C_e


	def getP(self,y_points):
		# y_points needs to be a list with arrays of coordinates in 2D space.
		#y_points = [[0.1,0.15],[0.3,0.2],[0.7,0.6]]
		ny = len(y_points)
		#P = np.zeros((ny, self.ne+1), dtype = float)
		P = np.zeros((ny, (self.a+1)*(self.b+1)), dtype = float)
		#x = np.array([0.334])
		self.el = self.V.element()  #get FEM basis function
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
		print("P:")
		print(P)
		return P


	def get_C_d(self,y_points,ld, sigd):
		#ld = 1e-16
		#sigd =1e-16
		#C_d = self.exponentiated_quadratic(y_points, y_points, lf=ld, sigf=sigd)
		#C_d = self.exponentiated_quadratic_log(y_points, y_points, l=ld, sig=sigd)
		C_d = self.matern_log(y_points, y_points, l=ld, sig=sigd)
		#print('C_d:')
		#print(C_d)
		self.C_d = C_d

		return C_d



	def getLogPostMultiple(self,params):
		rho = params[0]
		sigd = params[1]
		ld = params[2]
		y_valuesList=self.yVectors
		#y_points = np.transpose(np.atleast_2d(np.array(self.y_points)))
		y_points = np.atleast_2d(np.array(self.y_points))
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

		#samples = self.doMCMC([rho_est,sigd_est,ld_est])
		#print('MCMC: ',samples)
		#return result.x

		return rho_est,sigd_est,ld_est
		#return rho_est,sigd_est,0.2



	def computePosteriorMultipleY(self,y_points,y_values):
		""" here, y_values is a vector of different measurement sets. """

		C_e = self.get_C_e(len(y_points))
		P = self.getP(y_points)
		self.C_e = C_e
		pars = self.estimateHyperpar(y_points,self.yVectors)
		rho=pars[0]
		sigd = pars[1]
		ld = pars[2]

		#y_points = np.transpose(np.atleast_2d(np.array(y_points)))
		y_points = np.atleast_2d(np.array(y_points))
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

		self.nMC = 1
		posteriorGP = np.random.multivariate_normal(
			mean = u_mean_y, cov=C_u_y,
			size=self.nMC)
		self.posteriorGP = posteriorGP

		C_u_yDiag = np.sqrt(np.diagonal(C_u_y))
		u_y_sig = Function(self.V)
		u_y_sig.vector().set_local(C_u_yDiag.tolist())

		fig = plt.figure()
		c=plot(u_y_sig)
		plt.colorbar(c)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.scatter(np.transpose(self.y_points)[0],np.transpose(self.y_points)[1])
		fig.savefig("VarField_Posterior.pdf", bbox_inches='tight')
		plt.close(fig)


		u_y_mean = Function(self.V)
		u_y_mean.vector().set_local(u_mean_y.tolist())
		fig = plt.figure()
		c=plot(u_y_mean)
		plt.colorbar(c)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.scatter(np.transpose(self.y_points)[0],np.transpose(self.y_points)[1])
		
		fig.savefig("Mean_Posterior.pdf", bbox_inches='tight')
		plt.close(fig)


		return u_mean_y,posteriorGP




	def plotSolution(self):
		"""
		plots the complete solution nicely
		"""


		solArrayTrans = np.transpose(self.solArray)
		sigL = []
		muL = []
		DeltaL = []
		print("solArrayTrans:")
		print(np.shape(solArrayTrans))
		for point in solArrayTrans:
			mu, sig, Delta = self.doMC(point)
			sigL.append(sig)
			muL.append(mu)
			DeltaL.append(Delta)#


		# figSol = plt.figure(figsize=(6, 4), dpi=100)
		# c=plot(self.u)
		# #plt.colorbar(c)
		# plt.title("basic FEM without GP")
		# plt.show()
		X,Y,Z = self.x0.vector().get_local(), self.x1.vector().get_local(), self.u.vector().get_local()

		mean = np.zeros((self.a+1,self.b+1))

		for i in range((self.a+1)*(self.b+1)):
		#	print(X[i],Y[i])
			a,b = int(self.a*X[i]),int(self.b*Y[i])
			mean[a,b] = Z[i]

		x=np.linspace(0,1,self.a+1)
		y=np.linspace(0,1,self.b+1)
		Z = mean
		self.UMeanOutput.append(Z)
		self.UCutOutput.append(self.U_mean_cut)
		self.PriorOutput.append(self.SourceMean)
		self.UCutVarOutput.append(self.sig_cut)
		self.PriorVarOutput.append(self.sourceVariance)
		fig4 = plt.figure(figsize=(5.5,4), dpi=100)
		#ax = fig4.gca(projection='3d')
		ax = fig4.gca()

		#X2,Y2 = np.meshgrid(x,y)
		#ax.plot_surface(X2,Y2, np.transpose(Z), rstride=3, cstride=3, linewidth=1, antialiased=True,
		 #              cmap=cm.cividis)
		#ax.plot_wireframe(X2,Y2, np.transpose(Z),
		 #              cmap=cm.cividis)
		#c=plot(self.u)
		print("UmeanCut:")
		print(self.U_mean_cut)
		print(np.shape(self.U_mean_cut))
		cset2 = ax.contourf(x, y, np.transpose(Z), 100,cmap=cm.cividis)
	#	ax.plot(0.3*np.array(self.SourceMean),y,color="red", label="Source")
	#	plt.fill_betweenx(y, 0.3*((np.array(self.SourceMean) + 1.96*np.array(self.sourceVariance))),
	#			0.3*((np.array(self.SourceMean) - 1.96*np.array(self.sourceVariance))), color='red', alpha=0.15)

		ax.plot(x,0.7*np.ones(self.a+1),color="black",linestyle="--")

	#	ax.plot(x,0.7*np.ones(self.a+1),np	.array(self.U_mean_cut) + 1.96*np.array(sigL),color="green")
	#	ax.plot(x,0.7*np.ones(self.a+1),np.array(self.U_mean_cut) - 1.96*np.array(sigL),color="green")

		#ax.set_zlim(0,0.08)
		#ax.set_ylim(0,4)
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		ax.set_xticks(np.linspace(0,1,6))
		#ax.set_zlabel("$z$")
		#ax.view_init(24,-70)
		cbar = fig4.colorbar(cset2, ax=ax,  pad=0.14)

		ax2 = ax.twinx()
		ax2.plot(x,np.array(self.U_mean_cut),color="green", label="Pressure at y = 0.7")
		ax2.fill_between(x, (np.array(self.U_mean_cut) + 1.96*np.array(self.sig_cut)), (np.array(self.U_mean_cut) - 1.96*np.array(self.sig_cut)), color='green', alpha=0.15)
		ax2.plot(x,(np.array(self.U_mean_cut) + 1.96*np.array(sigL)),color='green', ls = "--",label="MC simulation")
		ax2.plot(x,(np.array(self.U_mean_cut) - 1.96*np.array(sigL)),color='green', ls = "--")
		#ax2.fill_between(x, (np.array(self.U_mean_cut) + 1.96*np.array(self.sig_cut)), (np.array(self.U_mean_cut) - 1.96*np.array(self.sig_cut)), color='green', alpha=0.15)
		ax.set_xlim(-0.25,1)
		ax.set_ylim(0,1)
		#ax2.set_ylim(-120,80)
		ax2.set_ylabel("Pressure at $y_c$",color='green', loc="top")
		ax2.tick_params(axis='y',labelcolor='green')
		ax2.set_ylim(-32,18)
		ax2.set_yticks(np.linspace(-15,15,4))

		ax3 = ax.twiny()
		ax3.plot(0.3*np.array(self.SourceMean),y,color="red", label="Source")
		ax3.fill_betweenx(y, 0.3*((np.array(self.SourceMean) + 1.96*np.array(self.sourceVariance))),
			0.3*((np.array(self.SourceMean) - 1.96*np.array(self.sourceVariance))), color='red', alpha=0.15)
		ax3.set_xlabel(r"$\rho \omega^2 \bar{U}$",color='red', loc="left")
		ax3.tick_params(axis='x',labelcolor='red')
		ax3.set_xticks(np.linspace(-10,10,3))
		ax3.set_xlim(-14,30)
	#	ax.set_zlim(-15,15)
		#ax.set_zticks(np.linspace(0,0.15,5))
		fig4.legend(loc="upper right",bbox_to_anchor=(1,1),bbox_transform=ax.transAxes)
		#plt.gca().set_aspect("equal")
		plt.tight_layout()
		align.xaxes(ax,0,ax3,0,0.2)
		fig4.savefig("SolutionCustom.pdf", bbox_inches='tight')

		

		plt.show()

solver = solverClass()
#solver.getDispGP()
solver.doFEM()



for i in range(2):
	solver.get_U_mean()
	solver.get_C_u()

fig3 = plt.figure()
for u in solver.solArray:
	plt.plot(solver.x_cut,u)

solArrayTrans = np.transpose(solver.solArray)
sigL = []
muL = []
DeltaL = []
print("solArrayTrans:")
print(np.shape(solArrayTrans))
for point in solArrayTrans:
	mu, sig, Delta = solver.doMC(point)
	sigL.append(sig)
	muL.append(mu)
	DeltaL.append(Delta)#


plt.plot(solver.x_cut,solver.U_mean_cut,label="mean",color="black",lw = 2.0)
#plt.plot(solver.x_cut, solver.C_uDiag, linestyle='-.',color = 'black',lw = 2.0,label='sig')
plt.plot(solver.x_cut, solver.U_mean_cut-1.96*np.array(sigL), linestyle='-.',color = 'black',lw = 2.0,label='2sig')
plt.plot(solver.x_cut, solver.U_mean_cut+1.96*np.array(sigL), linestyle='-.',color = 'black',lw = 2.0)
plt.title("SolutionGP")
plt.legend()
fig3.savefig("sol.pdf", bbox_inches='tight')
plt.figure()
for f in solver.fArray:#
	plt.plot(f)
plt.title("Forcing GP")
#plt.show()

solver.samplef(500)	


solver.y_points = [[0.1,0.15],[0.3,0.2],[0.7,0.6]]
solver.y_values = [15,5,5]

#add noise:
y_values_list = []
solver.no = 50
for i in range(solver.no):
	noise = np.random.normal(0,2.5e-3,len(solver.y_values))
	y_values_list.append(solver.y_values + noise)
solver.yVectors = y_values_list

solver.computePosteriorMultipleY(solver.y_points,solver.yVectors)
solver.plotSolution()


## stuff for the interactive jupyter notebooks:

# for f in np.linspace(1,3,3):
# 	#solver.f = f
# 	Params.factor = f
# 	c = 340
# 	solver.omega = 2* np.pi * solver.f
# 	solver.k = solver.omega / c
# 	#self.d_mean = 0.0000000001#0.05
# 	solver.g = solver.rho * solver.omega**2
# 	solver.doFEM()
# 	for i in range(2):
# 		solver.get_U_mean()
# 		solver.get_C_u()
# 	solArrayTrans = np.transpose(solver.solArray)
# 	sigL = []
# 	muL = []
# 	DeltaL = []
# 	print("solArrayTrans:")
# 	print(np.shape(solArrayTrans))
# 	for point in solArrayTrans:
# 		mu, sig, Delta = solver.doMC(point)
# 		sigL.append(sig)
# 		muL.append(mu)
# 		DeltaL.append(Delta)#

# 	solver.samplef(500)

# 	solver.plotSolution()

# DataOutput = [solver.UMeanOutput,solver.UCutOutput,solver.UCutVarOutput,solver.PriorOutput,solver.PriorVarOutput]
# with open(PIK, "wb") as f:
#     pickle.dump(DataOutput, f)