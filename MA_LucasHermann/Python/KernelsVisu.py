import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')

# Test data
n = 100
Xtest = np.linspace(-5,5, n).reshape(-1,1)



# covariance function (Kernel): squared exponential
#def kernel(a, b, param):#
# #   sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#    return np.exp(-.5 * (1/param) * sqdist)

def sqexp(xa, xb, lf, sigf):
	"""Exponentiated quadratic  with σ=1"""
	# L2 distance (Squared Euclidian)
	sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean') * (1/lf**2)
	return sigf**2 * np.exp(sq_norm)


def periodic(xa, xb, lf, sigf,p):
    sq_norm = -2*np.sin(np.pi * spatial.distance.cdist(xa, xb, 'euclidean')/p )**2 * (1/lf**2)
    #sq_norm = -2*np.sin(np.pi * (xa-xb)/p )**2 * (1/lf**2)
    return sigf**2 * np.exp(sq_norm)


def lin(xa, xb, sb, sv,c):
    return sb**2 + sv**2*(np.transpose(xa-c))*(xb-c)
	#return sb**2 + sv**2 * (xa * xb -c*(xa+xb))


def matern_log(xa, xb, l, sig):
    """ Matern Kernel: expects input parameters as log(par). """
    r = spatial.distance.cdist(xa, xb, 'euclidean')
    #return np.exp(2*sig) * np.exp(-1 * scipy.spatial.distance.cdist(xa, xb, 'euclidean') * np.exp(-1*l)) #nu = 1/2
    return np.exp(2*sig) * ((1 + np.sqrt(3) * r *np.exp(-1*l)) *  np.exp(-1 * np.sqrt(3) * r * np.exp(-1*l))   ) # nu = 3/2
    #return np.exp(2*sig) * ((1 + np.sqrt(5) * r *np.exp(-1*l) +  5*r*r/3*np.exp(-1*l)*np.exp(-1*l)  ) *  np.exp(-1 * np.sqrt(5) * r * np.exp(-1*l))   ) # nu = 5/2




l = 0.3
sig = 1.1
K_ss = sqexp(Xtest, Xtest, l, sig)
K_pe = periodic(Xtest, Xtest, 0.5, 2.5, 3)
K_lin = lin(Xtest, Xtest,1.2,0.8,1)
K_mat = matern_log(Xtest, Xtest,l=np.log(l),sig=np.log(sig))

fig = plt.figure(figsize=plt.figaspect(1))
ax0 = fig.add_subplot(2, 2, 1)
ax0.pcolormesh(K_ss)
ax0.set_xlabel('x')
ax0.set_ylabel("x'")
ax0.set_title('Squared Exponential')
ax0.axis('off') 

ax1 = fig.add_subplot(2, 2, 2)
ax1.pcolormesh(K_pe)
ax1.set_xlabel('x')
ax1.set_ylabel("x'")
ax1.set_title('Periodic')
ax1.axis('off')

ax3 = fig.add_subplot(2, 2, 3)
ax3.pcolormesh(K_lin)
ax3.set_xlabel('x')
ax3.set_ylabel("x'")
ax3.set_title('Linear')
ax3.axis('off')

ax4 = fig.add_subplot(2, 2, 4)
ax4.pcolormesh(K_mat)
ax4.set_xlabel('x')
ax4.set_ylabel("x'")
ax4.set_title('Matern')
ax4.axis('off')

#plt.show()


L = np.linalg.cholesky(K_ss + 1e-10*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,6)))
s2 = np.diag(K_ss)
stdv = np.sqrt(s2)



# Now let's plot the 3 sampled functions.
f=plt.figure(figsize=(12,4))
ax1 = plt.subplot(2,2,1)
plt.plot(Xtest, f_prior)
plt.gca().fill_between(Xtest.flat, 0-1.96*stdv, 0+1.96*stdv, color="green",alpha=0.2)
#plt.axis([0, 1, --0.25])
#plt.title('samples from the GP prior')
plt.grid()
plt.xlabel('x')
plt.ylabel('correlated output f(x)')


ax2 = plt.subplot(2,2,2)
L = np.linalg.cholesky(K_pe + 1e-10*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,3)))
s2 = np.diag(K_pe)
stdv = np.sqrt(s2)
plt.plot(Xtest, f_prior)
plt.gca().fill_between(Xtest.flat, 0-1.96*stdv, 0+1.96*stdv, color="green",alpha=0.2)
#plt.axis([0, 1, --0.25])
#plt.title('samples from the GP prior')
plt.grid()
plt.xlabel('x')
plt.ylabel('correlated output f(x)')


ax3 = plt.subplot(2,2,3)
L = np.linalg.cholesky(K_lin + 1e-10*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,6)))
s2 = np.diag(K_lin)
stdv = np.sqrt(s2)
plt.plot(Xtest, f_prior)
plt.gca().fill_between(Xtest.flat, 0-1.96*stdv, 0+1.96*stdv, color="green",alpha=0.2)
#plt.axis([0, 1, --0.25])
#plt.title('samples from the GP prior')
plt.grid()
plt.xlabel('x')
plt.ylabel('correlated output f(x)')


ax4 = plt.subplot(2,2,4)
L = np.linalg.cholesky(K_mat + 1e-10*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,6)))
s2 = np.diag(K_mat)
stdv = np.sqrt(s2)
plt.plot(Xtest, f_prior)
plt.gca().fill_between(Xtest.flat, 0-1.96*stdv, 0+1.96*stdv, color="green",alpha=0.2)
#plt.axis([0, 1, --0.25])
#plt.title('samples from the GP prior')
plt.grid()
plt.xlabel('x')
plt.ylabel('correlated output f(x)')



plt.show()


f=plt.figure(figsize=(12,4))
L = np.linalg.cholesky(K_lin + 1e-10*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,6)))
s2 = np.diag(K_lin)
stdv = np.sqrt(s2)
plt.plot(Xtest, f_prior)
plt.gca().fill_between(Xtest.flat, 0-1.96*stdv, 0+1.96*stdv, color="green",alpha=0.2)
#plt.axis([0, 1, --0.25])
#plt.title('samples from the GP prior')
plt.grid()
plt.xlabel('x')
plt.ylabel('correlated output f(x)')

plt.tight_layout()
plt.show()