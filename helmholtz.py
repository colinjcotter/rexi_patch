from firedrake import *
nref = 4
base = UnitSquareMesh(4, 4)
mh = MeshHierarchy(base,nref)
mesh = mh[-1]

degree = 1
V = FunctionSpace(mesh, "BDM", degree)
Q = FunctionSpace(mesh, "DG", degree-1)
W = V * Q

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

x, y = SpatialCoordinate(mesh)

eqn = inner(u,v)*dx - div(v)*p*dx + p*q*dx + div(u)*q*dx
L = q*exp(cos(pi*x)*cos(pi*y))*dx

bcs = [DirichletBC(W.sub(0), as_vector((0.,0.0)), "on_boundary")]

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

hprob = LinearVariationalProblem(eqn, L, w, bcs=bcs)
hsolver = LinearVariationalSolver(hprob, solver_parameters=mg_params)
hsolver.solve()
