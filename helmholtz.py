from firedrake import *

nref = 4
base = UnitSquareMesh(4, 4)
mh = MeshHierarchy(base,nref)
mesh = mh[-1]

degree = 1
V = VectorFunctionSpace(mesh, "BDM", degree)
Q = VectorFunctionSpace(mesh, "DG", degree-1)
W = V * Q

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

x, y = SpatialCoordinate(mesh)
f = exp(cos(pi*x)*cos(pi*y))

ur = u[0,:]
ui = u[1,:]
vr = v[0,:]
vi = v[1,:]
pr = p[0]
pi = p[1]
qr = q[0]
qi = q[1]

identity_bit = (inner(ur, vr) + inner(ui, vi)
                + pr*qr + pi*qi)*dx
L_bit = (- div(vr)*pr + div(ur)*qr
         - div(vi)*pi + div(ui)*qi)*dx

eqn = identity_bit + L_bit
rhs = qr*f*dx

w = Function(W)

lu_params = {
    "ksp_type":"preonly",
    "pc_type":"lu"
}

mg_params = {
    "snes_monitor": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    #"mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_codim": 0,
    "mg_levels_patch_pc_patch_construct_type": "vanka",
    "mg_levels_patch_pc_patch_local_type": "additive",
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps"
}

zs = Constant(0.)
zv = as_tensor([[zs, zs], [zs, zs]])

bcs = [DirichletBC(W.sub(0), zv, "on_boundary")]

hprob = LinearVariationalProblem(eqn, rhs, w, bcs=bcs)
hsolver = LinearVariationalSolver(hprob, solver_parameters=mg_params)
hsolver.solve()
