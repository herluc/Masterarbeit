import sys
from fenics import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import scipy
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

import scitools as st

import scitools.BoxField
#

class Params:
	dom_a = 0.0 #domain boundaries
	dom_b = 0.85


class solverClass:
	def __init__(self):
		self.dom_a =Params.dom_a# 0.0 #domain boundaries
		self.dom_b = Params.dom_b#0.85

		self.SourceBoundary = 1
		#print(Params.dom_b)
		#self.ne	= 5 #number of elements
		#self.mesh = IntervalMesh(self.ne,self.dom_a,self.dom_b) #define mesh
		self.a,self.b = 30,30
		self.mesh = UnitSquareMesh(self.a,self.b) #define mesh
		self.V = FunctionSpace(self.mesh, "Lagrange", 1) # Function space

		self.rho = 1.2
		self.f =340
		c = 340
		self.omega = 2* np.pi * self.f
		self.k = self.omega / c
		#self.d_mean = 0.0000000001#0.05
		self.g = self.rho * self.omega**2
		#self.g = rho * omega**2 * dis*np.sin(25*x[1])

		self.factor = 4
		factor = self.factor
		class MyExpression0(UserExpression):
			def eval(self, value, x):
				value[0] = 0.0001 *np.sin(factor*x[1])
			def value_shape(self):
				return ()
		self.f0 = MyExpression0()
		self.fArray = []
		self.f_bar_list= []
		self.solArray = []


	def exponentiated_quadratic_log(self, xa, xb, l, sig):
		""" expects input parameters as log(par). """
		return np.exp(2*sig) * np.exp(-0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * np.exp(-2*l))



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
		print("f_bar: links")
		f_bar = [x for x in b_mean.get_local() if x!=0.0]
		self.f_bar = f_bar
		print(f_bar)
		print(len(f_bar))
		# plt.figure
		# plt.plot(f_bar)
		# plt.show()
		# self.f_bar_list.append(f_bar)



		#self.bc.apply(b_mean)
		A = self.A
		solve(A,U_mean,b_mean)
		self.U_mean = U_mean


		#u = Function(self.V)
		#u.vector().set_local(U_mean.tolist()[0])

		#c = plot(u_mean,title="Prior mean")
		#plt.title("prior mean")
		#plt.colorbar(c)
		#plt.plot_surface(x_list,y_list,np.transpose(discrete_d))
		#plt.show()
		return U_mean



	def get_C_u(self):
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
		#self.C_uDiag = np.sqrt(np.diagonal(self.C_u))

		#C_f = C_f + 1e-10*ident
		#print("C_u:")
		#print(C_u)

		discrete_u = np.random.multivariate_normal(
			mean = self.U_mean , cov=C_u,
			size=1)
		##mean = 0.01*np.zeros(len(self.V.tabulate_dof_coordinates()))
		u = Function(self.V)
		u.vector().set_local(discrete_u.tolist()[0])

		# c = plot(u,title="Prior solution")
		# plt.title("prior sample")
		# plt.colorbar(c)
		#
		# plt.show()



		X = 0; Y = 1; Z = 0
		u_box = st.BoxField.dolfin_function2BoxField(u,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (0.0,0.7)
		x,uval,y_fixed,snapped = u_box.gridline(start, direction = X)
		self.x_cut = x
		#plt.figure()
		#plt.plot(x,uval)
		#plt.title("Prior sample cut")
		#plt.show()

		self.solArray.append(uval)

		return C_u
		#return c_u


	def get_C_f(self):
		ds = self.ds
		sb = self.SourceBoundary
		self.integratedTestF = np.array(assemble(Constant(1.0) * self.v * ds(sb)).get_local()) #* self.ne
		#print("intTestF:")
		#print(self.integratedTestF)
		y_list = np.expand_dims(np.linspace(0, 1, self.b+1), 1)
		x_list = np.expand_dims(np.linspace(0, 1, self.a+1), 1)

		c_f = self.exponentiated_quadratic_log(self.dof_coordinates, self.dof_coordinates, l=np.log(0.3), sig=np.log(12))
		self.c_f = c_f
		#print("c_f:")
		#print(np.shape(c_f))
		C_f = np.zeros(((self.a+1)*(self.b+1),(self.a+1)*(self.b+1)))
		#print('coords:')
		#print(self.coordinates)
		for i in range((self.a+1)*(self.b+1)):
			for j in range((self.a+1)*(self.b+1)):
				C_f[i,j] = self.integratedTestF[i] * c_f[i,j] * self.integratedTestF[j]
		#self.bc.apply(C_f)
		print("C_f:")
		print(C_f)
		print(np.shape(C_f))
		print("dof_Coord:")
		print(self.dof_coordinates)
		print(len(self.dof_coordinates))
		#plt.figure()
		#plot(self.mesh)
		#plt.title("mesh")
		#plt.show()
		ident = np.identity(len(self.V.tabulate_dof_coordinates()))
		mean = [0.0001 *np.sin(self.factor*y) for y in np.ones(len(self.V.tabulate_dof_coordinates())).tolist()]
		#print("mean f:")
		#print(mean)
		#np.sin*np.ones(len(self.V.tabulate_dof_coordinates()))
		discrete_d = np.random.multivariate_normal(
			mean = np.zeros(len(self.V.tabulate_dof_coordinates())) , cov=C_f,
			size=1)

		u = Function(self.V)

		u.vector().set_local(discrete_d.tolist()[0])
		#print(discrete_d.tolist())

		X = 0; Y = 1; Z = 0
		u_box = st.BoxField.dolfin_function2BoxField(u,self.mesh,(self.a,self.b),uniform_mesh=True)
		start = (1.0,0.0)
		x,uval,y_fixed,snapped = u_box.gridline(start, direction = Y)

		self.fArray.append(uval)
		self.x_f = x
		print("uval:")
		print(len(uval))
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
		#print("f_prior:")
		#print(discrete_d)


		plt.figure()
		c = plot(u)
		plt.colorbar(c)
		plt.title("forcing complete")
		plt.show()
		self.C_f = C_f
		return C_f


	def getDispGP(self):
		y_list = np.expand_dims(np.linspace(0, 1, self.b), 1) # Vector of poitns in the mesh
		d_mean = 0.01*np.ones(len(y_list))
		cov = self.exponentiated_quadratic_log(y_list, y_list, l=np.log(0.15), sig=np.log(0.001))
		discrete_d = np.random.multivariate_normal(
			mean = d_mean , cov=cov,
			size=1)
		#self.d_mean = d_mean
		plt.figure()
		plt.plot(y_list,np.transpose(discrete_d))
		plt.show()
		return discrete_d



	def doFEM(self):
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
		#print("debugBC:")
		#if bxL.inside(x, True):
	#		print('%s is on x = 0' % x)

		self.dof_coordinates = self.V.tabulate_dof_coordinates()
		n = self.V.dim()
		gdim = self.mesh.geometry().dim()
		dofmap = self.V.dofmap()
		dofs = dofmap.dofs()
	#	print(self.dof_coordinates)

		dof_x = self.V.tabulate_dof_coordinates().reshape((-1, gdim))
		x = dof_x[:,0]
		indices = np.where(np.logical_and(x > 0.26, x < 0.34))[0]
		xs = dof_x[indices]
		vals = U[indices]
		maxval = np.max(vals)
		vals/=3*maxval
		#for x, v in zip(xs, vals):
		#	print(x, v)

		#u = u.vector()
		#plt.plot(np.abs(U))
		#print("u:")
		#print(U.get_local())

		c=plot(self.u)
		plt.colorbar(c)
		plt.title("basic FEM without GP")
		plt.show()
		#plt.figure()
		#plt.plot(np.linspace(0,1,len(vals)),vals.tolist())
		#plt.show()
		#print("A:")
		#print(A.array())
		#print("b:")
		#print(b.get_local())

		self.A = A



solver = solverClass()
#solver.getDispGP()
solver.doFEM()

for i in range(3):
	solver.get_U_mean()
	solver.get_C_u()

plt.figure()
for u in solver.solArray:
	plt.plot(solver.x_cut,u)
plt.title("SolutionGP")
plt.figure()
for f in solver.fArray:#
	plt.plot(f)
plt.title("Forcing GP")
plt.show()

plt.show()
