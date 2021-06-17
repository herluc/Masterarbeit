import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage[utf8]{inputenc}')
from mpl_toolkits.mplot3d import Axes3D

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

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

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

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

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
