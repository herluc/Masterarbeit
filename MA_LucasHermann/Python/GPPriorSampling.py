# Imports
import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('darkgrid')
np.random.seed(42)
#


# Define the exponentiated quadratic kernel
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


# Sample from the Gaussian process distribution
nb_of_samples = 90  # Number of points in each function
number_of_functions = 5  # Number of functions to sample
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
ys2 = np.random.multivariate_normal(
    mean = mean , cov=Σ,
    size=number_of_functions)



#####Different approach with Cholesky decomposition
# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(Σ + 1e-15*np.eye(nb_of_samples))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
sigma = 1
f_prior = np.dot(L, np.random.normal(size=(nb_of_samples,number_of_functions), scale=sigma))

## add a mean function
f_priorT = np.transpose(f_prior)
for i in range(number_of_functions):
    f_priorT[i]+=mean
######################################



# Plot the sampled functions
plt.figure(figsize=(6, 4), dpi=100)
for i in range(number_of_functions):
    plt.plot(X, ys2[i], linestyle='-', marker='o', markersize=3)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$y = f(x)$', fontsize=13)
plt.title((
    '5 different function realizations at 41 points\n'
    'sampled from a Gaussian process with exponentiated quadratic kernel'))
plt.xlim([-4, 4])

#


#### For the different approach
plt.figure(figsize=(6, 4), dpi=100)
for i in range(number_of_functions):
    plt.plot(X, f_priorT[i], linestyle='-', marker='o', markersize=3)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$y = f(x)$', fontsize=13)
plt.title((
    '5 different function realizations at 41 points\n'
    'sampled from a Gaussian process with exponentiated quadratic kernel'))
plt.xlim([-4, 4])

plt.show()
