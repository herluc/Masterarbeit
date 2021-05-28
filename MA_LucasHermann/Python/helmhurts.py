# coding: utf-8
#
# Solving Helmhotz equation with FEniCS
# Author: Juan Luis Cano Rodríguez <juanlu@pybonacci.org>
# Inspired by: http://jasmcole.com/2014/08/25/helmhurts/
#
import sys
from dolfin import *


k = 50.0

## Problem data
E0 = Constant(0.0)
n = Constant(1.0)
k = Constant(k)  # 2.4 GHz / c

## Formulation
mesh = UnitSquareMesh(1000, 1000)
V = FunctionSpace(mesh, "Lagrange", 1)

E = TrialFunction(V)
v = TestFunction(V)

# Boundary conditions
point = Point(0.5, 0.5)
f = PointSource(V, point)

def E0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, E0, E0_boundary)

# Equation
a = ((k**2 / n**2) * inner(E, v) - inner(nabla_grad(E), nabla_grad(v))) * dx
L = Constant(0.0) * v * dx

# Assemble system
A, rhs = assemble_system(a, L, bc)
f.apply(rhs)

# Solve system
E = Function(V)
E_vec = E.vector()
solve(A, E_vec, rhs)

# Plot and export solution
plot(E, interactive=True)

file = File("helmhurts.pvd")
file << E
