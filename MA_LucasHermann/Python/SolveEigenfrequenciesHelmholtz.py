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

class Params:
    # parameter class is introduced for Fenics Expression classes which couldn't 
    # access the parameters in their surrounding class otherwise.
	dom_a = 0.0 #domain boundaries
	dom_b = 0.85

	factor = 1 # factor for sine wave: 2sin(factor * pi *x)
    
class solverClass:
    def __init__(self):
        self.dom_a =Params.dom_a# 0.0 #domain boundaries, fethcing from params class
        self.dom_b = Params.dom_b#0.85

        self.SourceBoundary = 1 #determines on which boundary part the source is applied. 0 left, 1 right

        self.a,self.b = 30,30 #no of elements in x,y direction
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
                # different shapes to choose from
                val = 20 *np.sin(factor*np.pi*x[1])
                val = 0.00001 *np.sin(2*np.pi*x[1]) + 0.000005 *np.sin(4*np.pi*x[1])
                val = 0.00001 *np.sin(Params.factor*np.pi*x[1]) + 0.000005 *np.sin(2*Params.factor*np.pi*x[1])
                if x[0] >(1.0-1e-8):
                    value[0] = val
                    value[0] = 0.000003 #overwrite sine using a constant function
                else:
                    value[0] = 0.0 #only the boundary is non-zero. evry other point is zero.

            def value_shape(self):
                return ()

        self.f0 = MyExpression1()
        self.f1 = MyExpression1() # f0 and f1 are both used. needs to be changed to a single one since they're the same


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

        



    def getMeanSquarePress(self):
        bar_p_sq = 0
        for p in self.u.vector().get_local():
            val = np.sqrt(np.abs(p*p))
            bar_p_sq+=val
            print(bar_p_sq)
        bar_p_sq = bar_p_sq / len(self.u.vector().get_local())
        print(bar_p_sq)
        return bar_p_sq
       

    

    def doFEM(self):
        
        # Define variational problem

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        f0 = self.f0

        boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1	, 0)

        tol = 1e-14
        class BoundaryX_L(SubDomain):
            self.tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 0, tol)# and (x[1] < 0.3)

        class BoundaryX_R(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 1, tol) and (x[1] < 0.3)

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
        print("dofCords:")
        print(self.dof_coordinates)
        #print(np.random.shuffle(self.dof_coordinates))
        # dof_x = self.V.tabulate_dof_coordinates().reshape((-1, gdim))
        # x = dof_x[:,0]
        # indices = np.where(np.logical_and(x > 0.26, x < 0.34))[0]
        # xs = dof_x[indices]
        # vals = U[indices]
        # maxval = np.max(vals)
        # vals/=3*maxval


        #for x, v in zip(xs, vals):
        #	print(x, v)

        #u = u.vector()
        #plt.plot(np.abs(U))
        #print("u:")
        #print(U.get_local())

       # c=plot(self.u)
       # plt.colorbar(c)
       # plt.title("basic FEM without GP")
       # plt.show()
        #plt.figure()
        #plt.plot(np.linspace(0,1,len(vals)),vals.tolist())
        #plt.show()
        #print("A:")
        #print(A.array())
        #print("b:")
        #print(b.get_local())

        self.A = A


        self.x0 = interpolate(Expression("x[0]",degree=1),self.V)
        self.x1 = interpolate(Expression("x[1]",degree=1),self.V)
        print("x0 local:")
        print(self.x0.vector().get_local())
        #for i in range(31*31):
        #	print(self.x0.vector().get_local()[i],self.x1.vector().get_local()[i])

solver = solverClass()
solver.doFEM()
solver.getMeanSquarePress()
FreqRange = np.linspace(1,600,200)
FRF = []
for f in FreqRange:
    solver.f = f
    c = 340
    solver.omega = 2* np.pi * solver.f
    solver.k = solver.omega / c
    #self.d_mean = 0.0000000001#0.05
    solver.g = solver.rho * solver.omega**2
    #self.g = rho * omega**2 * dis*np.sin(25*x[1])

    solver.doFEM()

    FRF.append(solver.getMeanSquarePress())

fig = plt.figure()
plt.plot(FreqRange,FRF,color='black')

plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('Mean square pressure [Pa]')
fig.savefig("Eigenfrequencies.pdf", bbox_inches='tight')

plt.show()

