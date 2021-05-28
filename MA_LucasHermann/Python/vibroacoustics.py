import sys
from fenics import *

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import scipy
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
#

class Params:
	dom_a = 0.0 #domain boundaries
	dom_b = 0.85


class solverClass:
	def __init__(self):
		self.dom_a =Params.dom_a# 0.0 #domain boundaries
		self.dom_b = Params.dom_b#0.85
		print(Params.dom_b)
		self.ne	= 100 #number of elements
		self.mesh = IntervalMesh(self.ne,self.dom_a,self.dom_b) #define mesh
		self.V = FunctionSpace(self.mesh, "Lagrange", 1) # Function space

		rho = 1.2
		self.f = 600
		c = 340
		omega = 2* np.pi * self.f
		self.k = omega / c	
		dis = 0.01
		self.g = rho * omega**2 * dis

	def doFEM(self):
		# Define variational problem

		self.u = TrialFunction(self.V)
		self.v = TestFunction(self.V)

		#boundary_markers = FacetFunction('size_t', self.mesh)
		boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)
		class BoundaryX_L(SubDomain):
			self.tol = 1E-14
			def inside(self, x, on_boundary):
				return on_boundary and near(x[0], Params.dom_b, 1E-14)
		boundary_markers.set_all(0) # all markers default to zero.
		bxL = BoundaryX_L()
		bxL.mark(boundary_markers,1) # boundary bxL is marked as "1"
		#g = rho * omega**2 * u
		ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers) #ds (contrary to dS) describes only the boundaries. so: using ds(0) would mean using all boundaries. if we set markers, we can pick single boudnaries out of the pool.

		a = dot(grad(self.u), grad(self.v))*dx - self.k**2 * dot(self.u,self.v)*dx #variational Problem
		L = self.v*self.g*ds(1) # L is nonzero only on theboudnary marked with "1". if ds(0) would be used, all two boundaries would be nonzero.

		A = assemble(a)
		b = assemble(L)
		self.u = Function(self.V)
		U = self.u.vector()
		solve(A, U, b)
		#u = u.vector()
		plt.plot(np.abs(U))
		plt.show()
		print("A:")
		print(A.array())
		print("b:")
		print(b.get_local())

solver = solverClass()
solver.doFEM()