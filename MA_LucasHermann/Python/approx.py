import sympy as sym

def least_squares(f, psi, Omega):
	N = len(psi) - 1
	A = sym.zeros(N+1, N+1)
	b = sym.zeros(N+1, 1)
	x = sym.Symbol(’x’)
	for i in range(N+1):
		for j in range(i, N+1):
			A[i,j] = sym.integrate(psi[i]*psi[j],
				(x, Omega[0], Omega[1]))
			A[j,i] = A[i,j]
		b[i,0] = sym.integrate(psi[i]*f, (x, Omega[0], Omega[1]))
	c = A.LUsolve(b)
	# Note: c is a sympy Matrix object, solution is in c[:,0]
	u = 0
	for i in range(len(psi)):
		u += c[i,0]*psi[i]
	return u, c
