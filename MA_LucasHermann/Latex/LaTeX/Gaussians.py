import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, random

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
Y2=Y
X2=X
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 2.])
Sigma = np.array([[ 1. , -0.7], [-0.7,  1.5]])
mu = np.array([0., 0])
Sigma = np.array([[ 1 , 0.05], [0.05,  1]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(colors)

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    N_s = 10


    f=plt.figure(figsize=(8,6))

    ax1 = plt.subplot(2,2,1)
    samples = np.zeros((N_s,n))
    t = np.array(range(1,n+1))
    mu = np.array([0., 0])
    Sigma = np.array([[ 1 , 0.05], [0.05,  1]])
    for i in np.arange(N_s):
        samples[i,:] = np.random.multivariate_normal(mean = mu, cov = Sigma)
        plt.plot(t,samples[i,:],'-o',ms=5,color=colors[i])
    plt.gca().set_title('Weak correlation')
    plt.xlabel('dimension')
    plt.ylabel('sampled value')
    plt.grid()
    plt.ylim(-2,2)


    ax2 = plt.subplot(2,2,3)
    x= samples[:,0]
    y = samples[:,1]
    X,Y = np.meshgrid(np.linspace(x.min()-1,x.max()+1,100), np.linspace(y.min()-1,y.max()+1,100))
    rv = stats.multivariate_normal(mean = mu, cov = Sigma)
    Z = rv.pdf(np.dstack((X,Y)))
    cset = ax2.contour(X, Y, Z,cmap=cm.cividis)
    for i in np.arange(N_s):
        #samples[i,:] = np.random.multivariate_normal(mean = mu, cov = Sigma)
        plt.plot(samples[i,0],samples[i,1],'-o',ms=5,color=colors[i])
    plt.xlabel('dim. 1')
    plt.ylabel('dim. 2')
    plt.grid()

    ax3 = plt.subplot(2,2,2)
    samples = np.zeros((N_s,n))
    t = np.array(range(1,n+1))
    mu = np.array([0., 0])
    Sigma = np.array([[ 1 , 0.95], [0.95,  1]])
    for i in np.arange(N_s):
        samples[i,:] = np.random.multivariate_normal(mean = mu, cov = Sigma)
        plt.plot(t,samples[i,:],'-o',ms=5,color=colors[i])
    plt.gca().set_title('Strong correlation')
    plt.ylim(-2,2)
    plt.xlabel('dimension')
    plt.ylabel('sampled value')
    plt.grid()


    ax4 = plt.subplot(2,2,4)
    x= samples[:,0]
    y = samples[:,1]
    X,Y = np.meshgrid(np.linspace(x.min()-1,x.max()+1,100), np.linspace(y.min()-1,y.max()+1,100))
    rv = stats.multivariate_normal(mean = mu, cov = Sigma)
    Z = rv.pdf(np.dstack((X,Y)))
    cset = ax4.contour(X, Y, Z,cmap=cm.cividis)
    for i in np.arange(N_s):
        #samples[i,:] = np.random.multivariate_normal(mean = mu, cov = Sigma)
        plt.plot(samples[i,0],samples[i,1],'-o',ms=5,color=colors[i])
    plt.xlabel('dim. 1')
    plt.ylabel('dim. 2')
    plt.show()




    f2=plt.figure(figsize=(8,6))

    ax1 = plt.subplot(1,2,1)
    mu = np.array([0,0,0,0,0,0])
    D = mu.shape[0]
    samples = np.zeros((5,D))

    tmp = np.sort(random.rand(D))[:,None]
    tmp2 = tmp**np.arange(5)
    Sigma = 5*np.dot(tmp2,tmp2.T) + 0.005*np.eye(D)
    for i in np.arange(5):
        samples[i,:] = np.random.multivariate_normal(mean = mu, cov = Sigma)
        plt.plot(tmp,samples[i,:],'-o',ms=5,color=colors[i])

    plt.plot(tmp,np.diagonal(Sigma),'-o',ms=5,color='black',alpha=0.5)
    plt.plot(tmp,-np.diagonal(Sigma),'-o',ms=5,color='black',alpha=0.5)
    plt.gca().set_title('6 dimensions')
    plt.grid()




    ax2 = plt.subplot(1,2,2)
    D=100
    mu = np.zeros((D,1))[:,0]
    samples = np.zeros((5,D))

    tmp = np.sort(random.rand(D))[:,None]
    tmp2 = tmp**np.arange(5)
    Sigma = 5*np.dot(tmp2,tmp2.T) + 0.00005*np.eye(D)
    for i in np.arange(5):
        samples[i,:] = np.random.multivariate_normal(mean = mu, cov = Sigma)
        plt.plot(tmp,samples[i,:],color=colors[i])
    plt.plot(tmp,np.diagonal(Sigma),color='black',alpha=0.5,label='Variance')
    plt.plot(tmp,-np.diagonal(Sigma),color='black',alpha=0.5)
    plt.gca().set_title('100 dimensions')
    plt.grid()


    plt.show()






    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

quit()


print("Z:")
print(Z)

y0 = 30

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.cividis)



cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.cividis)


# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.15)
ax.set_zticks(np.linspace(0,0.15,5))
ax.view_init(21,-45)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel("$z$")
plt.tight_layout()

fig.savefig("Gaussians.pdf", bbox_inches='tight')
plt.show()

fig2d=plt.figure(figsize=(6, 4), dpi=100)
print(Z[:,y0])
print(X[0])
plt.plot(X2,Z[y0,:],color="black")
plt.xlabel("$x$")
plt.ylabel("z")
plt.grid()
plt.tight_layout()
fig2d.savefig("Gaussians2d.pdf", bbox_inches='tight')

plt.show()

print(X[0][40])
fig3=plt.figure(figsize=(6, 4), dpi=100)
plt.contour(X, Y, Z, zdir='z', cmap=cm.cividis)
plt.plot(X[0],Y2[y0]*np.ones(60),color="black")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.ylim(-1,4)
plt.grid()
plt.tight_layout()
fig3.savefig("Gaussians2dCut.pdf", bbox_inches='tight')
plt.show()


fig4 = plt.figure(figsize=(6, 4), dpi=100)
ax = fig4.gca(projection='3d')
cset2 = ax.contour(X, Y, Z, 10,zdir='z', offset = 0.0,cmap=cm.cividis)
ax.plot(X2,Y2[y0]*np.ones(60),Z[y0,:],color="black")
plt.plot(X[0],Y2[y0]*np.ones(60),np.zeros(60),color="black",linestyle="--")
ax.set_zlim(0,0.08)
#ax.set_ylim(0,4)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel("$z$")
ax.view_init(24,-70)
plt.tight_layout()
fig4.savefig("Gaussians3dCut.pdf", bbox_inches='tight')
plt.show()
