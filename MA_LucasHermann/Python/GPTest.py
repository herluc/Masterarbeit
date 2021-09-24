import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')
from scipy import spatial
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF

# Test data
n = 50
Xtest = np.linspace(0, 1, n).reshape(-1,1)



# covariance function (Kernel): squared exponential
#def kernel(a, b, param):#
# #   sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#    return np.exp(-.5 * (1/param) * sqdist)

def kernel(xa, xb, lf, sigf):
	"""Exponentiated quadratic  with σ=1"""
	# L2 distance (Squared Euclidian)
	sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
	return sigf**2 * np.exp(sq_norm)

l = 0.3
sig = 1.1
K_ss = kernel(Xtest, Xtest, l, sig)


# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-10*np.eye(n))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,6)))

#s2 = np.diag(K_ss) - np.sum(L**2, axis=0)
s2 = np.diag(K_ss)
stdv = np.sqrt(s2)

# Now let's plot the 3 sampled functions.
f=plt.figure(figsize=(7,2.5))
ax1 = plt.subplot(1,2,1)
plt.plot(Xtest, f_prior)
plt.gca().fill_between(Xtest.flat, 0-1.96*stdv, 0+1.96*stdv, color="green",alpha=0.2)
#plt.axis([0, 1, --0.25])
#plt.title('samples from the GP prior')
plt.grid()
plt.xlabel('x')
plt.ylabel('correlated output f(x)')
plt.tight_layout()
#plt.show()


# Noiseless training data
Xtrain = np.array([0.2,0.6,0.8]).reshape(3,1)
Xtrain = np.array([0.2,0.5]).reshape(2,1)
#Xtrain = np.array([0.5]).reshape(1,1)
#Xtrain = np.array([0.2]).reshape(1,1)
ytrain = np.sin(6*np.pi*Xtrain)

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, l, sig)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, l, sig)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))


# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,6)))

ax2 = plt.subplot(1,2,2)
plt.plot(Xtrain, ytrain, 'o', ms=8,color='black')
plt.plot(Xtest, f_post)
plt.gca().fill_between(Xtest.flat, mu-1.96*stdv, mu+1.96*stdv, color="black",alpha=0.2)
plt.plot(Xtest, mu, lw=2,color='black')
#plt.axis([0,1, -1,1])
#plt.title('Three samples from the GP posterior')
plt.xlabel('x')
plt.ylabel('correlated output f(x)')
plt.grid()
plt.tight_layout()
plt.show()



def pHighlyNonlin(x):
    #return 3*np.sin(20*x) - 22*x**2 + 5*x
	return np.sin(8*x) + 2*x**2 +(0.5*x)


def pLin(x):
    #return 3*np.sin(20*x) - 22*x**2 + 5*x
	return (0.5*x**2)

X = np.linspace(0, 1, 100, endpoint=True)
X_train = np.random.uniform(0, 1 ,20)
F = pHighlyNonlin(X_train)
z = np.polyfit(X_train, F, 2)
p = np.poly1d(z)



f=plt.figure(figsize=(7,2.5))
ax1 = plt.subplot(1,2,2)
plt.scatter(X_train,F)
plt.plot(X,p(X),color='blue',alpha=0.6,lw=3,label='deg. 2 poly. regression')

#X=np.asarray(X).reshape(-1,1)
X_train=np.asarray(X_train).reshape(-1,1)
F=np.asarray(F).reshape(-1,1)
kernel = 0.2 * RBF(0.2)
gpr = GaussianProcessRegressor(random_state=0)
gpr.fit(X_train,F)
y_pred_gaussian,sigma = gpr.predict(X.reshape(-1,1), return_std=True)

plt.plot(X, y_pred_gaussian[:,0], color="green",ls='--',lw=3,label='GP regression')
#plt.fill_between(X, y_pred_gaussian[:,0]-1.96*1000*sigma, y_pred_gaussian[:,0]+1.96*1000*sigma, color="black",alpha=0.2)
plt.plot(X,pHighlyNonlin(X),color='black',lw=0.7,label='ground truth')
plt.grid()
plt.title('y=sin(8*x) + 2*x**2 +(0.5*x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

ax2 = plt.subplot(1,2,1)
X_train = np.random.uniform(0, 1 ,20)
F = pLin(X_train)
z = np.polyfit(X_train, F, 2)
p = np.poly1d(z)
plt.scatter(X_train,F)
plt.plot(X,p(X),color='blue',alpha=0.6,lw=3,label='deg. 2 poly. regression')
X_train=np.asarray(X_train).reshape(-1,1)
F=np.asarray(F).reshape(-1,1)
kernel = 0.2 * RBF(0.2)
gpr = GaussianProcessRegressor(random_state=0)
gpr.fit(X_train,F)
y_pred_gaussian,sigma = gpr.predict(X.reshape(-1,1), return_std=True)

plt.plot(X, y_pred_gaussian[:,0], color="green",ls='--',alpha=1,lw=3,label='GP regression')
#plt.fill_between(X, y_pred_gaussian[:,0]-1.96*1000*sigma, y_pred_gaussian[:,0]+1.96*1000*sigma, color="black",alpha=0.2)
plt.plot(X,pLin(X),color='black',lw=0.7,label='ground truth')
plt.title('y=0.5x')
plt.grid()
plt.legend()
plt.tight_layout()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


fig = plt.figure(figsize=plt.figaspect(1))
ax1 = fig.add_subplot(1,2, 1)
ax1.pcolormesh(K_ss)
ax1.set_xlabel('x')
ax1.set_ylabel("x'")
ax1.set_title('Prior')
ax1.axis('off') 


ax0 = fig.add_subplot(1,2, 2)
ax0.pcolormesh(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
ax0.set_xlabel('x')
ax0.set_ylabel("x'")
ax0.set_title('Posterior')
ax0.axis('off') 
plt.show()