import sympy as sym
import numpy as np

def least_squares(f, psi, Omega, symbolic=True):
	N = len(psi) - 1
	A = sym.zeros(N+1, N+1)
	b = sym.zeros(N+1, 1)
	x = sym.Symbol('x')
	for i in range(N+1):
		for j in range(i, N+1):
			integrand = psi[i]*psi[j]
			if symbolic:
				I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
			if not symbolic or isinstance(I, sym.Integral):
				# Could not integrate symbolically, use numerical int.
				integrand = sym.lambdify([x], integrand, 'mpmath')
				I = mpmath.quad(integrand, [Omega[0], Omega[1]])
			A[i,j] = A[j,i] = I
			
		integrand = psi[i]*f
		if symbolic:
			I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
		if not symbolic or isinstance(I, sym.Integral):
			# Could not integrate symbolically, use numerical int.
			integrand = sym.lambdify([x], integrand, 'mpmath')
			I = mpmath.quad(integrand, [Omega[0], Omega[1]])
		b[i,0] = I
	if symbolic:
		c = A.LUsolve(b) # symbolic solve
		# c is a sympy Matrix object, numbers are in c[i,0]
		c = [sym.simplify(c[i,0]) for i in range(c.shape[0])]
	else:
		c = mpmath.lu_solve(A, b) # numerical solve
		c = [c[i,0] for i in range(c.rows)]
	u = sum(c[i]*psi[i] for i in range(len(psi)))
	return u, c
	
	

def regression(f, psi, points):
	N = len(psi) - 1
	m = len(points)
	# Use numpy arrays and numerical computing
	B = np.zeros((N+1, N+1))
	d = np.zeros(N+1)
	# Wrap psi and f in Python functions rather than expressions
	# so that we can evaluate psi at points[i]
	x = sym.Symbol('x')
	psi_sym = psi # save symbolic expression
	psi = [sym.lambdify([x], psi[i]) for i in range(N+1)]
	f = sym.lambdify([x], f)
	for i in range(N+1):
		for j in range(N+1):
			B[i,j] = 0
			for k in range(m+1):
				B[i,j] += psi[i](points[k])*psi[j](points[k])
		d[i] = 0
		for k in range(m+1):
			d[i] += psi[i](points[k])*f(points[k])
	c = np.linalg.solve(B, d)
	u = sum(c[i]*psi_sym[i] for i in range(N+1))
	return u, c


	
def comparison_plot(f, u, Omega, filename='tmp.pdf'):
	x = sym.Symbol('x')
	f = sym.lambdify([x], f, modules="numpy")
	u = sym.lambdify([x], u, modules="numpy")
	resolution = 401 # no of points in plot
	xcoor = linspace(Omega[0], Omega[1], resolution)
	exact = f(xcoor)
	approx = u(xcoor)
	plot(xcoor, approx)
	hold('on')
	plot(xcoor, exact)
	legend(['approximation', 'exact'])
	savefig(filename)
	
x = sym.Symbol('x')
f = 10*(x-1)**2 - 1
psi = [1, x]
Omega = [1, 2]
m_values = [2-1, 8-1, 64-1]
# Create m+3 points and use the inner m+1 points
for m in m_values:
	points = np.linspace(Omega[0], Omega[1], m+3)[1:-1]
	u, c = regression(f, psi, points)
	comparison_plot(
		f, u, Omega,
		filename='parabola_by_regression_%d' % (m+1),
		points=points,
		points_legend='%d interpolation points' % (m+1),
		legend_loc='upper left')

