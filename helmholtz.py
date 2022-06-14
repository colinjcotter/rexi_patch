from firedrake import *
n = 20
mesh = PeriodicUnitSquareMesh(20, 20)

degree = 1
V = FunctionSpace(mesh, "BDM", degree)
Q = FunctionSpace(mesh, "DG", degree-1)
W = V * Q

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

x, y = SpatialCoordinate(mesh)

eqn = inner(u,v)*dx - div(v)*p*dx + p*q*dx + div(u)*q*dx
L = q*exp(cos(pi*x)*cos(pi*y))*dx

w = Function(W)

lu_params = {
    "ksp_type":"preonly",
    "pc_type":"lu"
}

hprob = LinearVariationalProblem(eqn, L, w)
hsolver = LinearVariationalSolver(hprob, solver_parameters=lu_params)
hsolver.solve()
