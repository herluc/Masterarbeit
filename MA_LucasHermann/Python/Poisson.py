import sys
from fenics import *
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.gridspec as gridspec
import seaborn as sns
#sns.set_style('darkgrid')
import numpy as np
import scipy
np.random.seed(22)
from bisect import bisect_left
#




##################################
# GP Part
##################################


# Define the exponentiated quadratic kernel
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


# Sample from the Gaussian process distribution
nb_of_samples = 100  # Number of points in each function
number_of_functions = 42  # Number of functions to sample
# Independent variable samples
X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
print(X)
print(len(X))
Σ = exponentiated_quadratic(X, X)  # Kernel of data points

# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
ys = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=Σ,
    size=number_of_functions)


# Draw samples from the prior at our data points.
# Assume a mean function
pars = [0.3, -0.2] # polynomial parameters
mean = np.transpose(0 + X*pars[0] + X**2 * pars[1])[0]
mean = 6.0*np.ones(nb_of_samples)
ys2 = np.random.multivariate_normal(
    mean = mean , cov=Σ,
    size=number_of_functions)


pars = [0.3, -0.2] # polynomial parameters
mean = np.transpose(0 + X*pars[0] + X**2 * pars[1])[0]
mean = 6.0*np.ones(nb_of_samples)
def draw_FEM_samples(x_vals):
	Σ = exponentiated_quadratic(x_vals, x_vals)  # Kernel of data points
	discrete_f = np.random.multivariate_normal(
   		mean = mean , cov=Σ,
    	size=1)
	return discrete_f




######Different approach with Cholesky decomposition
## Get cholesky decomposition (square root) of the
## covariance matrix
#L = np.linalg.cholesky(Σ + 1e-4*np.eye(nb_of_samples))
## Sample 3 sets of standard normals for our test points,
## multiply them by the square root of the covariance matrix
#sigma = 1
#f_prior = np.dot(L, np.random.normal(size=(nb_of_samples,number_of_functions), scale=sigma))
#
### add a mean function
#f_priorT = np.transpose(f_prior)
#for i in range(number_of_functions):
#    f_priorT[i]+=mean
######################################



# Plot the sampled functions
#plt.figure(figsize=(6, 4), dpi=100)
#for i in range(number_of_functions):
#    plt.plot(X, ys2[i], linestyle='-', marker='o', markersize=3)
#plt.xlabel('$x$', fontsize=13)
#plt.ylabel('$y = f(x)$', fontsize=13)
#plt.title((
#    '5 different function realizations at 41 points\n'
#    'sampled from a Gaussian process with exponentiated quadratic kernel'))
#plt.xlim([-4, 4])

#


#### For the different approach
#plt.figure(figsize=(6, 4), dpi=100)
#for i in range(number_of_functions):
#    plt.plot(X, f_priorT[i], linestyle='-', marker='o', markersize=3)
#plt.xlabel('$x$', fontsize=13)
#plt.ylabel('$y = f(x)$', fontsize=13)
#plt.title((
#    '5 different function realizations at 41 points\n'
#    'sampled from a Gaussian process with exponentiated quadratic kernel'))
#plt.xlim([-4, 4])

#plt.show()



##################################
# FEM Part
##################################

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns index of the closest number in a list to a single number.
    """
    pos = bisect_left(myList, myNumber)
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return pos
    else:
       return pos-1
    return pos





def take_closest2(xlist, myNumber):
    """
    Assumes myList is sorted. Returns index of the closest number in a list to a single number.
    """
    print(xlist)
    print(myNumber)
    pos = np.where(xlist == myNumber)
    print(pos)
    pos = pos[0]
    print(pos)
    pos = pos[0]
    print(pos)
    return pos




# Create mesh and define function space
#mesh = UnitSquareMesh(32, 32)

ne = 100 #number of elements
dom_a = -4.0
dom_b = 4.0
#f_vals = np.ones(ne)
#f_vals = f_vals *(-6.0)
h_vec = np.linspace(dom_a,dom_b,ne)
h_vec_sol = np.linspace(dom_a,dom_b,ne+1)
mesh = IntervalMesh(ne,dom_a,dom_b)
bbt = mesh.bounding_box_tree()
gdim = mesh.geometry().dim()
V = FunctionSpace(mesh, "Lagrange", 1)
dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))
coordinates = mesh.coordinates()
print(coordinates)
print('coordsLen',len(coordinates))
print('dofs_x', dofs_x)
print(len(dofs_x))
coordinates = (coordinates[1:] + coordinates[:-1])/2
print(coordinates)
print('coordsLen',len(coordinates))


xvecTest = []
solution = []
for i in range(number_of_functions):

	# Define Dirichlet boundary (x = 0 or x = 1)
	def boundary(x):
		return x[0] < (dom_a + DOLFIN_EPS) or x[0] > dom_b - DOLFIN_EPS

	# Define boundary condition
	u0 = Constant(0.0)
	bc = DirichletBC(V, u0, boundary)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	#f = Constant(-6.0)

	#f_vals = f_priorT[i] #cholesky approach
	f_vals = draw_FEM_samples(coordinates)
	f_vals = f_vals[0]
	print('coords')
	print(len(coordinates))
	print('f_vals')
	print(len(f_vals))
	f_vals = ys2[i] # numpy approach

	class MyExpression0(UserExpression):
		def eval(self, value, x):
			pos = take_closest(h_vec,x[0])
			value[0] = f_vals[pos]
			xvecTest.append(x[0])
		def value_shape(self):
			return ()


	class DoGP(UserExpression):
		def eval(self, value, x):
			collisions1st = bbt.compute_first_entity_collision(Point(x[0])) # find index of cell which contains point.
			value[0] = f_vals[collisions1st] # f is constant in a cell. therefore there are 100 cells and 100 values for f.
		def value_shape(self):
			return ()
	#f = MyExpression0()
	f = DoGP()


	#f = Expression("-6.0 * x[0]", degree=2)
	a = dot(grad(u), grad(v))*dx
	L = f*v*dx


	u = Function(V)
	solve(a == L, u, bc)

	solution.append(u.vector().get_local())

##mean curve:####
# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
	return x[0] < (dom_a + DOLFIN_EPS) or x[0] > dom_b - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

#f = Constant(mean)


f = Expression("6.0", degree=0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx


u = Function(V)
solve(a == L, u, bc)

meancurve= u.vector().get_local()
#################


# Plot solution and mesh
print(solution)
print(len(solution))


plt.figure(figsize=(6, 4), dpi=100)
plt.plot(dofs_x, np.transpose(solution), linestyle='-')
plt.plot(dofs_x, meancurve, linestyle='-', color = 'black', lw = 4.0)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$u$', fontsize=13)
plt.title(('solution'))
plt.xlim([-4, 4])

plt.show()

#plot(u)
#plot(mesh)
#plt.show()




