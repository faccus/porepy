"""
Tests for components of the Ad machinery. Specifically, the tests cover:
    * Projection operators between collections of grids and individual grids
    * Projections between grids and mortar grids (again also covering collections)
    * Ad representations of various grid-based operators (divergence etc).
    * Ad representation of variables.
    * The forward Ad machinery as implemented in pp.ad.Ad_array.

"""

import unittest

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import Ad_array

## Tests of the Ad grid operators.


@pytest.mark.parametrize("scalar", [True, False])
def test_subdomain_projections(scalar):
    """Test of subdomain projections. Both face and cell restriction and prolongation.

    Test three specific cases:
        1. Projections generated by passing a bucket and a list of grids are identical
        2. All projections for all grids (individually) in a simple bucket.
        3. Combined projections for list of grids.
    """

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 2])
    NC = gb.num_cells()
    NF = gb.num_faces()
    Nd = gb.dim_max()

    if scalar:
        proj_dim = 1
    else:
        NC *= Nd
        NF *= Nd
        proj_dim = Nd

    grid_list = np.array([g for g, _ in gb])
    proj = pp.ad.SubdomainProjections(grids=grid_list, nd=proj_dim)

    cell_start = np.cumsum(np.hstack((0, np.array([g.num_cells for g in grid_list]))))
    face_start = np.cumsum(np.hstack((0, np.array([g.num_faces for g in grid_list]))))

    # Helper method to get indices for sparse matrices
    def _mat_inds(nc, nf, grid_ind, scalar, Nd, cell_start, face_start):
        cell_inds = np.arange(cell_start[grid_ind], cell_start[grid_ind + 1])
        face_inds = np.arange(face_start[grid_ind], face_start[grid_ind + 1])
        if scalar:
            data_cell = np.ones(nc)
            row_cell = np.arange(nc)
            data_face = np.ones(nf)
            row_face = np.arange(nf)
            col_cell = cell_inds
            col_face = face_inds
        else:
            data_cell = np.ones(nc * Nd)
            row_cell = np.arange(nc * Nd)
            data_face = np.ones(nf * Nd)
            row_face = np.arange(nf * Nd)
            col_cell = pp.fvutils.expand_indices_nd(cell_inds, Nd)
            col_face = pp.fvutils.expand_indices_nd(face_inds, Nd)
        return row_cell, col_cell, data_cell, row_face, col_face, data_face

    # Test projection of one fracture at a time for the full set of grids
    for g in grid_list:

        ind = _list_ind_of_grid(grid_list, g)

        nc, nf = g.num_cells, g.num_faces

        num_rows_cell = nc
        num_rows_face = nf
        if not scalar:
            num_rows_cell *= Nd
            num_rows_face *= Nd

        row_cell, col_cell, data_cell, row_face, col_face, data_face = _mat_inds(
            nc, nf, ind, scalar, Nd, cell_start, face_start
        )

        known_cell_proj = sps.coo_matrix(
            (data_cell, (row_cell, col_cell)), shape=(num_rows_cell, NC)
        ).tocsr()
        known_face_proj = sps.coo_matrix(
            (data_face, (row_face, col_face)), shape=(num_rows_face, NF)
        ).tocsr()

        assert _compare_matrices(proj.cell_restriction(g), known_cell_proj)
        assert _compare_matrices(proj.cell_prolongation(g), known_cell_proj.T)
        assert _compare_matrices(proj.face_restriction(g), known_face_proj)
        assert _compare_matrices(proj.face_prolongation(g), known_face_proj.T)

    # Project between the full grid and both 1d grids (to combine two grids)
    g1, g2 = gb.grids_of_dimension(1)
    rc1, cc1, dc1, rf1, cf1, df1 = _mat_inds(
        g1.num_cells,
        g1.num_faces,
        _list_ind_of_grid(grid_list, g1),
        scalar,
        Nd,
        cell_start,
        face_start,
    )
    rc2, cc2, dc2, rf2, cf2, df2 = _mat_inds(
        g2.num_cells,
        g2.num_faces,
        _list_ind_of_grid(grid_list, g2),
        scalar,
        Nd,
        cell_start,
        face_start,
    )

    # Adjust the indices of the second grid, we will stack the matrices.
    rc2 += rc1.size
    rf2 += rf1.size
    num_rows_cell = (g1.num_cells + g2.num_cells) * proj_dim
    num_rows_face = (g1.num_faces + g2.num_faces) * proj_dim

    known_cell_proj = sps.coo_matrix(
        (np.hstack((dc1, dc2)), (np.hstack((rc1, rc2)), np.hstack((cc1, cc2)))),
        shape=(num_rows_cell, NC),
    ).tocsr()
    known_face_proj = sps.coo_matrix(
        (np.hstack((df1, df2)), (np.hstack((rf1, rf2)), np.hstack((cf1, cf2)))),
        shape=(num_rows_face, NF),
    ).tocsr()

    assert _compare_matrices(proj.cell_restriction([g1, g2]), known_cell_proj)
    assert _compare_matrices(proj.cell_prolongation([g1, g2]), known_cell_proj.T)
    assert _compare_matrices(proj.face_restriction([g1, g2]), known_face_proj)
    assert _compare_matrices(proj.face_prolongation([g1, g2]), known_face_proj.T)


@pytest.mark.parametrize("scalar", [True, False])
def test_mortar_projections(scalar):
    # Test of mortar projections between mortar grids and standard subdomain grids.
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 2])
    Nd = gb.dim_max()

    if scalar:
        proj_dim = 1
    else:
        proj_dim = Nd

    NC = gb.num_cells() * proj_dim
    NF = gb.num_faces() * proj_dim
    NMC = gb.num_mortar_cells() * proj_dim

    g0 = gb.grids_of_dimension(2)[0]
    g1, g2 = gb.grids_of_dimension(1)
    g3 = gb.grids_of_dimension(0)[0]

    mg01 = gb.edge_props((g0, g1), "mortar_grid")
    mg02 = gb.edge_props((g0, g2), "mortar_grid")

    mg13 = gb.edge_props((g1, g3), "mortar_grid")
    mg23 = gb.edge_props((g2, g3), "mortar_grid")

    ########
    # First test projection between all grids and all interfaces
    grid_list = np.array([g0, g1, g2, g3])
    edge_list = [(g0, g1), (g0, g2), (g1, g3), (g2, g3)]

    proj = pp.ad.MortarProjections(grids=grid_list, edges=edge_list, gb=gb, nd=proj_dim)

    cell_start = proj_dim * np.cumsum(
        np.hstack((0, np.array([g.num_cells for g in grid_list])))
    )
    face_start = proj_dim * np.cumsum(
        np.hstack((0, np.array([g.num_faces for g in grid_list])))
    )

    f0 = np.hstack(
        (
            sps.find(mg01.mortar_to_primary_int(nd=proj_dim))[0],
            sps.find(mg02.mortar_to_primary_int(nd=proj_dim))[0],
        )
    )
    f1 = sps.find(mg13.mortar_to_primary_int(nd=proj_dim))[0]
    f2 = sps.find(mg23.mortar_to_primary_int(nd=proj_dim))[0]

    c1 = sps.find(mg01.mortar_to_secondary_int(nd=proj_dim))[0]
    c2 = sps.find(mg02.mortar_to_secondary_int(nd=proj_dim))[0]
    c3 = np.hstack(
        (
            sps.find(mg13.mortar_to_secondary_int(nd=proj_dim))[0],
            sps.find(mg23.mortar_to_secondary_int(nd=proj_dim))[0],
        )
    )

    rows_higher = np.hstack((f0, f1 + face_start[1], f2 + face_start[2]))
    cols_higher = np.arange(NMC)
    data = np.ones(NMC)

    proj_known_higher = sps.coo_matrix(
        (data, (rows_higher, cols_higher)), shape=(NF, NMC)
    ).tocsr()

    assert _compare_matrices(proj_known_higher, proj.mortar_to_primary_int)
    assert _compare_matrices(proj_known_higher, proj.mortar_to_primary_avg)
    assert _compare_matrices(proj_known_higher.T, proj.primary_to_mortar_int)
    assert _compare_matrices(proj_known_higher.T, proj.primary_to_mortar_avg)

    rows_lower = np.hstack((c1 + cell_start[1], c2 + cell_start[2], c3 + cell_start[3]))
    cols_lower = np.arange(NMC)
    data = np.ones(NMC)

    proj_known_lower = sps.coo_matrix(
        (data, (rows_lower, cols_lower)), shape=(NC, NMC)
    ).tocsr()
    assert _compare_matrices(proj_known_lower, proj.mortar_to_secondary_int)

    # Also test block matrices for the sign of mortar projections.
    # This is a diagonal matrix with first -1, then 1.
    # If this test fails, something is fundentally wrong.
    vals = np.array([])
    for e in edge_list:
        mg = gb.edge_props(e, "mortar_grid")
        sz = int(np.round(mg.num_cells / 2))
        if not scalar:
            sz *= Nd
        vals = np.hstack((vals, -np.ones(sz), np.ones(sz)))

    known_sgn_mat = sps.dia_matrix((vals, 0), shape=(NMC, NMC))
    assert _compare_matrices(known_sgn_mat, proj.sign_of_mortar_sides)


@pytest.mark.parametrize("scalar", [True, False])
def test_boundary_condition(scalar):
    """Test of boundary condition representation."""
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 2])

    grid_list = np.array([g for g, _ in gb])

    Nd = gb.dim_max()
    key = "foo"

    # Start of all faces
    face_start = np.cumsum(np.hstack((0, np.array([g.num_faces for g in grid_list]))))

    # Build values of known values (to be filled during assignment of bcs)
    if scalar:
        known_values = np.zeros(gb.num_faces())
    else:
        known_values = np.zeros(gb.num_faces() * Nd)
        # If vector problem, all faces have Nd numbers
        face_start *= Nd

    # Loop over grids, assign values, keep track of assigned values
    for g, d in gb:
        grid_ind = _list_ind_of_grid(grid_list, g)
        if scalar:
            values = np.random.rand(g.num_faces)
        else:
            values = np.random.rand(g.num_faces * Nd)

        d[pp.PARAMETERS] = {key: {"bc_values": values}}

        # Put face values in the right place in the vector of knowns
        face_inds = np.arange(face_start[grid_ind], face_start[grid_ind + 1])
        known_values[face_inds] = values

    # Ad representation of the boundary conditions. Parse.
    bc = pp.ad.BoundaryCondition(key, grid_list)
    val = bc.parse(gb)

    assert np.allclose(val, known_values)


## Tests of Ad operators


def test_ad_variable_wrappers():
    # Tests that the wrapping of Ad variables, including previous iterates
    # and time steps, are carried out correctly.
    # See also test_variable_combinations, which specifically tests evaluation of
    # variables in a setting of multple variables, including merged variables.

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 2])

    state_map = {}
    iterate_map = {}

    state_map_2, iterate_map_2 = {}, {}

    var = "foo"
    var2 = "bar"

    mortar_var = "mv"

    def _compare_ad_objects(a, b):
        va, ja = a.val, a.jac
        vb, jb = b.val, b.jac

        assert np.allclose(va, vb)
        assert ja.shape == jb.shape
        d = ja - jb
        if d.data.size > 0:
            assert np.max(np.abs(d.data)) < 1e-10

    for g, d in gb:
        if g.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        d[pp.PRIMARY_VARIABLES] = {var: {"cells": num_dofs}}

        val_state = np.random.rand(g.num_cells * num_dofs)
        val_iterate = np.random.rand(g.num_cells * num_dofs)

        d[pp.STATE] = {var: val_state, pp.ITERATE: {var: val_iterate}}
        state_map[g] = val_state
        iterate_map[g] = val_iterate

        # Add a second variable to the 2d grid, just for the fun of it
        if g.dim == 2:
            d[pp.PRIMARY_VARIABLES][var2] = {"cells": 1}
            val_state = np.random.rand(g.num_cells)
            val_iterate = np.random.rand(g.num_cells)
            d[pp.STATE][var2] = val_state
            d[pp.STATE][pp.ITERATE][var2] = val_iterate
            state_map_2[g] = val_state
            iterate_map_2[g] = val_iterate

    for e, d in gb.edges():
        mg = d["mortar_grid"]
        if mg.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": num_dofs}}

        val_state = np.random.rand(mg.num_cells * num_dofs)
        val_iterate = np.random.rand(mg.num_cells * num_dofs)

        d[pp.STATE] = {mortar_var: val_state, pp.ITERATE: {mortar_var: val_iterate}}
        state_map[e] = val_state
        iterate_map[e] = val_iterate

    dof_manager = pp.DofManager(gb)
    eq_manager = pp.ad.EquationManager(gb, dof_manager)

    # Manually assemble state and iterate
    true_state = np.zeros(dof_manager.num_dofs())
    true_iterate = np.zeros(dof_manager.num_dofs())

    # Also a state array that differs from the stored iterates
    double_iterate = np.zeros(dof_manager.num_dofs())

    for (g, v) in dof_manager.block_dof:
        inds = dof_manager.grid_and_variable_to_dofs(g, v)
        if v == var2:
            true_state[inds] = state_map_2[g]
            true_iterate[inds] = iterate_map_2[g]
            double_iterate[inds] = 2 * iterate_map_2[g]
        else:
            true_state[inds] = state_map[g]
            true_iterate[inds] = iterate_map[g]
            double_iterate[inds] = 2 * iterate_map[g]

    grid_list = [
        gb.grids_of_dimension(2)[0],
        *gb.grids_of_dimension(1),
        gb.grids_of_dimension(0)[0],
    ]

    # Generate merged variables via the EquationManager.
    var_ad = eq_manager.merge_variables([(g, var) for g in grid_list])

    # Check equivalence between the two approaches to generation.

    # Check that the state is correctly evaluated.
    inds_var = np.hstack(
        [dof_manager.grid_and_variable_to_dofs(g, var) for g in grid_list]
    )
    assert np.allclose(
        true_iterate[inds_var], var_ad.evaluate(dof_manager, true_iterate).val
    )

    # Check evaluation when no state is passed to the parser, and information must
    # instead be glued together from the GridBucket
    assert np.allclose(true_iterate[inds_var], var_ad.evaluate(dof_manager).val)

    # Evaluate the equation using the double iterate
    assert np.allclose(
        2 * true_iterate[inds_var], var_ad.evaluate(dof_manager, double_iterate).val
    )

    # Represent the variable on the previous time step. This should be a numpy array
    prev_var_ad = var_ad.previous_timestep()
    prev_evaluated = prev_var_ad.evaluate(dof_manager)
    assert isinstance(prev_evaluated, np.ndarray)
    assert np.allclose(true_state[inds_var], prev_evaluated)

    # Also check that state values given to the ad parser are ignored for previous
    # values
    assert np.allclose(
        prev_evaluated, prev_var_ad.evaluate(dof_manager, double_iterate)
    )

    ## Next, test edge variables. This should be much the same as the grid variables,
    # so the testing is less thorough.
    # Form an edge variable, evaluate this
    edge_list = [e for e, _ in gb.edges()]
    var_edge = eq_manager.merge_variables([(e, mortar_var) for e in edge_list])

    edge_inds = np.hstack(
        [dof_manager.grid_and_variable_to_dofs(e, mortar_var) for e in edge_list]
    )
    assert np.allclose(
        true_iterate[edge_inds], var_edge.evaluate(dof_manager, true_iterate).val
    )

    # Finally, test a single variable; everything should work then as well
    g = gb.grids_of_dimension(2)[0]
    v1 = eq_manager.variable(g, var)
    v2 = eq_manager.variable(g, var2)

    ind1 = dof_manager.grid_and_variable_to_dofs(g, var)
    ind2 = dof_manager.grid_and_variable_to_dofs(g, var2)

    assert np.allclose(true_iterate[ind1], v1.evaluate(dof_manager, true_iterate).val)
    assert np.allclose(true_iterate[ind2], v2.evaluate(dof_manager, true_iterate).val)

    v1_prev = v1.previous_timestep()
    assert np.allclose(true_state[ind1], v1_prev.evaluate(dof_manager, true_iterate))


@pytest.mark.parametrize(
    "grids", [[pp.CartGrid([4, 1])], [pp.CartGrid([4, 1]), pp.CartGrid([3, 1])]]
)
@pytest.mark.parametrize(
    "variables",
    [["foo"], ["foo", "bar"]],
)
def test_variable_combinations(grids, variables):
    # Test combinations of variables, and merged variables, on different grids.
    # Main check is if Jacobian matrices are of the right size.

    # Make GridBucket, populate with necessary information
    gb = pp.GridBucket()
    gb.add_nodes(grids)
    for g, d in gb:
        d[pp.STATE] = {}
        d[pp.PRIMARY_VARIABLES] = {}
        for var in variables:
            d[pp.PRIMARY_VARIABLES].update({var: {"cells": 1}})
            d[pp.STATE][var] = np.random.rand(g.num_cells)

    # Ad boilerplate
    dof_manager = pp.DofManager(gb)
    eq_manager = pp.ad.EquationManager(gb, dof_manager)

    # Standard Ad variables
    ad_vars = [eq_manager.variable(g, var) for g in grids for var in variables]
    # Merge variables over all grids
    merged_vars = []
    for var in variables:
        gv = [(g, var) for g in grids]
        merged_vars.append(eq_manager.merge_variables(gv))

    # First check of standard variables. If this fails, something is really wrong
    for g in grids:
        d = gb.node_props(g)
        for var in ad_vars:
            if g == var._g:
                expr = var.evaluate(dof_manager)
                # Check that the size of the variable is correct
                assert np.allclose(expr.val, d[pp.STATE][var._name])
                # Check that the Jacobian matrix has the right number of columns
                assert expr.jac.shape[1] == dof_manager.num_dofs()

    # Next, check that merged variables are handled correctly.
    for var in merged_vars:
        expr = var.evaluate(dof_manager)
        vals = []
        for sub_var in var.sub_vars:
            vals.append(gb.node_props(sub_var._g, pp.STATE)[sub_var._name])

        assert np.allclose(expr.val, np.hstack([v for v in vals]))
        assert expr.jac.shape[1] == dof_manager.num_dofs()

    # Finally, check that the size of the Jacobian matrix is correct when combining
    # variables (this will cover both variables and merged variable with the same name,
    # and with different name).
    for g in grids:
        for var in ad_vars:
            nc = var.size()
            cols = np.arange(nc)
            data = np.ones(nc)
            for mv in merged_vars:
                nr = mv.size()

                # The variable must be projected to the full set of grid for addition
                # to be meaningful. This requires a bit of work.
                sv_size = np.array([sv.size() for sv in mv.sub_vars])
                mv_grids = [sv._g for sv in mv.sub_vars]
                ind = mv_grids.index(var._g)
                offset = np.hstack((0, np.cumsum(sv_size)))[ind]
                rows = offset + np.arange(nc)
                P = pp.ad.Matrix(sps.coo_matrix((data, (rows, cols)), shape=(nr, nc)))

                eq = eq = mv + P * var
                expr = eq.evaluate(dof_manager)
                # Jacobian matrix size is set according to the dof manager,
                assert expr.jac.shape[1] == dof_manager.num_dofs()


@pytest.mark.parametrize(
    "operator_class, constructor_kwargs",
    [
        (
            "DiagonalJacobianFunction",
            ({"multipliers": [-1.0, 0.0]}, {"multipliers": [0.0, 2.0 * 1.816]}),
        ),
        ("Function", ({}, {})),
    ],
)
def test_operator_functions(operator_class, constructor_kwargs):
    """
    1. Test: test of if a certain operator function can solve the following nonlinear problem

    Find x, y s.t.
        y-x = 0
        y**2 + x**2 - r = 0
    for r in [1., 1.25, 1.5, 1.75]
    with initial guess
        x_0 = y_0 = r / (1 + 0.2**2) + 0.1

    2. Test: test of dimensions compatibility for a mixed-dimensional domain with mortar grids
    but without mortar variables
    """
    # The name of the tested class is given as a test parameter in form of a string.
    # In order to get a reference to the actual type, look for the class name among the
    # attributes of the module object `pp.ad`
    operator_cls = getattr(pp.ad, operator_class)

    ############ 1. Test: single domain for convergence test
    gb = pp.GridBucket()
    g = pp.CartGrid([2, 2])
    gb.add_nodes([g])
    gb.compute_geometry()

    # Functions for equations
    line = lambda x, y: y - x

    radii = np.array([1.0 + i * 0.25 for i in range(gb.num_cells())])
    circle_ = lambda x, y: x**2 + y**2

    # creating variables
    for g, d in gb:
        d[pp.STATE] = {}
        d[pp.STATE][pp.ITERATE] = {}
        d[pp.PRIMARY_VARIABLES] = {}

        d[pp.PRIMARY_VARIABLES].update({"x": {"cells": 1}})
        d[pp.PRIMARY_VARIABLES].update({"y": {"cells": 1}})

    dof_manager = pp.DofManager(gb)
    eq_manager = pp.ad.EquationManager(gb, dof_manager)

    for g, _ in gb:
        _ = [eq_manager.variable(g, "x"), eq_manager.variable(g, "y")]
        x = eq_manager.merge_variables([(g, "x")])
        y = eq_manager.merge_variables([(g, "y")])

    # Function for setting and resetting initial values. Returns the global initial state
    def reset_values(gb, eq):
        # initial guess is intersection with line with slope s, with a shift in y=x direction
        s = 0.2
        shift = 0.1 * radii / radii
        x_init = radii / np.sqrt(1 + s**2) + shift
        y_init = radii / np.sqrt(1 + s**2) + shift

        for g, d in gb:
            d[pp.STATE]["x"] = np.copy(x_init)
            d[pp.STATE][pp.ITERATE]["x"] = np.copy(x_init)
            d[pp.STATE]["y"] = np.copy(y_init)
            d[pp.STATE][pp.ITERATE]["y"] = np.copy(y_init)

        # Get current global vector using the identity
        # will be used as starting point for iterations
        identity = lambda x: x
        identity_op = pp.ad.Function(identity, "identity")
        eq.equations = {"identity_x": identity_op(x), "identity_y": identity_op(y)}
        _, b0 = eq.assemble()

        return (-1) * np.copy(b0)

    # Setting equations with approximating operators
    Z = reset_values(gb, eq_manager)

    line_a = operator_cls(
        **constructor_kwargs[0], func=line, name="line_a", is_vector_func=False
    )(x, y)

    circle_a_ = operator_cls(
        **constructor_kwargs[1], func=circle_, name="circle_a", is_vector_func=False
    )(x, y)
    circle_a = circle_a_ - radii**2

    eq_manager.equations = {"line": line_a, "circle": circle_a}
    iterations = 0
    max_iter = 1000
    res_norm = None

    # solving the nonlinear problem (see docstring: Test 1)
    for i in range(max_iter + 1):
        iterations = i

        A, b = eq_manager.assemble()

        res_norm = np.linalg.norm(b)
        if res_norm <= 1.0e-10:
            break

        dz = sps.linalg.spsolve(A, b)

        Z += dz
        dof_manager.distribute_variable(Z)
        dof_manager.distribute_variable(Z, to_iterate=True)

    assert iterations < max_iter

    ############ mD domain for test of dimensions
    # this grid has in total 200 cell DOFs
    # 88 cells in the 2D domain,
    # 4 cells in the 1D domain,
    # 8x2 cells on the Mortar grid
    coord_point = np.array([[0.2, 0.8], [0.5, 0.5]])
    indices_point = np.array([[0], [1]])
    domain = pp.SquareDomain([1, 1])
    fracture_network = pp.FractureNetwork2d(coord_point, indices_point, domain)
    mesh_args = {"mesh_size_frac": 0.3}
    mdg = fracture_network.mesh(mesh_args)
    mdg.compute_geometry()

    # creating variables
    for g, d in mdg:
        d[pp.STATE] = {}
        d[pp.STATE][pp.ITERATE] = {}
        d[pp.PRIMARY_VARIABLES] = {}

        d[pp.PRIMARY_VARIABLES].update({"x": {"cells": 1}})
        d[pp.PRIMARY_VARIABLES].update({"y": {"cells": 1}})

        d[pp.STATE]["x"] = np.ones(g.num_cells)
        d[pp.STATE][pp.ITERATE]["x"] = np.ones(g.num_cells)
        d[pp.STATE]["y"] = np.ones(g.num_cells)
        d[pp.STATE][pp.ITERATE]["y"] = np.ones(g.num_cells)

    # Then for the edges
    for _, d in mdg.edges():
        d[pp.PRIMARY_VARIABLES] = {"mortar_x": {"cells": 1}}
        d[pp.PRIMARY_VARIABLES].update({"mortar_y": {"cells": 1}})

    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    x = eq_manager.merge_variables([(g, "x") for g, _ in mdg])
    y = eq_manager.merge_variables([(g, "y") for g, _ in mdg])

    initial_values = np.ones(dof_manager.num_dofs())
    dof_manager.distribute_variable(initial_values)

    multiply = lambda x, y: x * y
    op = operator_cls(**constructor_kwargs[0], func=multiply, name="bb-multip")(x, y)

    op_ad = op.evaluate(dof_manager)
    # test of dimension
    assert op_ad.val.shape == (92,)  # 92 is the total number of subdomain cells
    assert op_ad.jac.shape == (
        92,
        200,
    )  # 200 is the total number of DOFS ( 92x2 + 8x2 )
    # test of values
    assert np.allclose(1.0, op_ad.val)


def test_ad_discretization_class():
    # Test of the mother class of all discretizations (pp.ad.Discretization)

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 2])

    grid_list = [g for g, _ in gb]
    sub_list = grid_list[:2]

    # Make two Mock discretizaitons, with different keywords
    key = "foo"
    sub_key = "bar"
    discr = _MockDiscretization(key)
    sub_discr = _MockDiscretization(sub_key)

    # Ad wrappers
    # This is mimicks the old init of Discretization, before it was decided to
    # make that class semi-ABC. Still checks the wrap method
    discr_ad = pp.ad.Discretization()
    discr_ad.grids = grid_list
    discr_ad._discretization = discr
    pp.ad._ad_utils.wrap_discretization(discr_ad, discr, grid_list)
    sub_discr_ad = pp.ad.Discretization()
    sub_discr_ad.grids = sub_list
    sub_discr_ad._discretization = sub_discr
    pp.ad._ad_utils.wrap_discretization(sub_discr_ad, sub_discr, sub_list)

    # values
    known_val = np.random.rand(len(grid_list))
    known_sub_val = np.random.rand(len(sub_list))

    # Assign a value to the discretization matrix, with the right key
    for vi, g in enumerate(grid_list):
        d = gb.node_props(g)
        d[pp.DISCRETIZATION_MATRICES] = {key: {"foobar": known_val[vi]}}

    # Same with submatrix
    for vi, g in enumerate(sub_list):
        d = gb.node_props(g)
        d[pp.DISCRETIZATION_MATRICES].update({sub_key: {"foobar": known_sub_val[vi]}})

    # Compare values under parsing. Note we need to pick out the diagonal, due to the
    # way parsing make block matrices.
    assert np.allclose(known_val, discr_ad.foobar.parse(gb).diagonal())
    assert np.allclose(known_sub_val, sub_discr_ad.foobar.parse(gb).diagonal())


## Below are helpers for tests of the Ad wrappers.


def _compare_matrices(m1, m2):
    if isinstance(m1, pp.ad.Matrix):
        m1 = m1._mat
    if isinstance(m2, pp.ad.Matrix):
        m2 = m2._mat
    if m1.shape != m2.shape:
        return False
    d = m1 - m2
    if d.data.size > 0:
        if np.max(np.abs(d.data)) > 1e-10:
            return False
    return True


def _list_ind_of_grid(grid_list, g):
    for i, gl in enumerate(grid_list):
        if g == gl:
            return i

    raise ValueError("grid is not in list")


class _MockDiscretization:
    def __init__(self, key):
        self.foobar_matrix_key = "foobar"
        self.not_matrix_keys = "failed"

        self.keyword = key


class AdArrays(unittest.TestCase):
    """Tests for the implementation of the main Ad array class,
    that is, the functionality needed for the forward Ad operations.
    """

    def test_add_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(-10, 0)
        c = a + b
        self.assertTrue(c.val == -9 and c.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)
        self.assertTrue(b.val == -10 and b.jac == 0)

    def test_add_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a + b
        self.assertTrue(np.allclose(c.val, 13) and np.allclose(c.jac, 4.0))
        self.assertTrue(a.val == 4 and np.allclose(a.jac, 1.0))
        self.assertTrue(b.val == 9 and b.jac == 3)

    def test_add_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a + b
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and np.allclose(a.jac, 2))
        self.assertTrue(b == 3)

    def test_add_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b + a
        self.assertTrue(np.allclose(c.val, 6) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_sub_two_scalars(self):
        a = Ad_array(1, 0)
        b = Ad_array(3, 0)
        c = a - b
        self.assertTrue(c.val == -2 and c.jac == 0)
        self.assertTrue(a.val == 1 and a.jac == 0)
        self.assertTrue(b.val == 3 and a.jac == 0)

    def test_sub_two_ad_variables(self):
        a = Ad_array(4, 1.0)
        b = Ad_array(9, 3)
        c = a - b
        self.assertTrue(np.allclose(c.val, -5) and np.allclose(c.jac, -2))
        self.assertTrue(a.val == 4 and np.allclose(a.jac, 1.0))
        self.assertTrue(b.val == 9 and b.jac == 3)

    def test_sub_var_with_scal(self):
        a = Ad_array(3, 2)
        b = 3
        c = a - b
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac, 2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_sub_scal_with_var(self):
        a = Ad_array(3, 2)
        b = 3
        c = b - a
        self.assertTrue(np.allclose(c.val, 0) and np.allclose(c.jac, -2))
        self.assertTrue(a.val == 3 and a.jac == 2)
        self.assertTrue(b == 3)

    def test_mul_scal_ad_scal(self):
        a = Ad_array(3, 0)
        b = Ad_array(2, 0)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == 0)
        self.assertTrue(a.val == 3 and a.jac == 0)
        self.assertTrue(b.val == 2 and b.jac == 0)

    def test_mul_ad_var_ad_scal(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, 0)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == 6)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b.val == 2 and b.jac == 0)

    def test_mul_ad_var_ad_var(self):
        a = Ad_array(3, 3)
        b = Ad_array(2, -4)
        c = a * b
        self.assertTrue(c.val == 6 and c.jac == -6)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b.val == 2 and b.jac == -4)

    def test_mul_ad_var_scal(self):
        a = Ad_array(3, 3)
        b = 3
        c = a * b
        self.assertTrue(c.val == 9 and c.jac == 9)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b == 3)

    def test_mul_scar_ad_var(self):
        a = Ad_array(3, 3)
        b = 3
        c = b * a
        self.assertTrue(c.val == 9 and c.jac == 9)
        self.assertTrue(a.val == 3 and a.jac == 3)
        self.assertTrue(b == 3)

    def test_mul_ad_var_mat(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = x * A
        sol = np.array([30, 36, 42])
        jac = np.diag([3, 2, 1]) * A

        self.assertTrue(np.all(f.val == sol) and np.all(f.jac == jac))
        self.assertTrue(
            np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))
        )
        self.assertTrue(np.all(A == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    def test_advar_mul_vec(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        A = np.array([1, 3, 10])
        f = x * A
        sol = np.array([1, 6, 30])
        jac = np.diag([3, 6, 10])

        self.assertTrue(np.all(f.val == sol) and np.all(f.jac == jac))
        self.assertTrue(
            np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))
        )

    def test_advar_m_mul_vec_n(self):
        x = Ad_array(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
        vec = np.array([1, 2])
        R = sps.csc_matrix(np.array([[1, 0, 1], [0, 1, 0]]))
        y = R * x
        z = y * vec
        Jy = np.array([[1, 0, 3], [0, 2, 0]])
        Jz = np.array([[1, 0, 3], [0, 4, 0]])
        self.assertTrue(np.all(y.val == [4, 2]))
        self.assertTrue(np.sum(y.full_jac().A - Jy) == 0)
        self.assertTrue(np.all(z.val == [4, 4]))
        self.assertTrue(np.sum(z.full_jac().A - Jz) == 0)

    def test_mul_sps_advar(self):
        J = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        x = Ad_array(np.array([1, 2, 3]), J)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        f = A * x

        self.assertTrue(np.all(f.val == [14, 32, 50]))
        self.assertTrue(np.all(f.jac == A * J.A))

    def test_mul_advar_vectors(self):
        Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
        Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), Ja)
        b = Ad_array(np.array([1, 1, 1]), Jb)
        A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        f = A * a + b

        self.assertTrue(np.all(f.val == [15, 33, 51]))
        self.assertTrue(np.sum(f.full_jac() != A * Ja + Jb) == 0)
        self.assertTrue(
            np.sum(Ja != sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]])))
            == 0
        )
        self.assertTrue(
            np.sum(Jb != sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
            == 0
        )

    def test_power_advar_scalar(self):
        a = Ad_array(2, 3)
        b = a**2
        self.assertTrue(b.val == 4 and b.jac == 12)

    def test_power_advar_advar(self):
        a = Ad_array(4, 4)
        b = Ad_array(-8, -12)
        c = a**b
        jac = -(2 + 3 * np.log(4)) / 16384
        self.assertTrue(np.allclose(c.val, 4**-8) and np.allclose(c.jac, jac))

    def test_rpower_advar_scalar(self):
        # Make an Ad_array with value 2 and derivative 3.
        a = Ad_array(2, 3)
        b = 2**a
        self.assertTrue(b.val == 4 and b.jac == 12 * np.log(2))

        c = 2 ** (-a)
        self.assertTrue(c.val == 1 / 4 and c.jac == 2 ** (-2) * np.log(2) * (-3))

    def test_rpower_advar_vector_scalar(self):
        J = sps.csc_matrix(np.array([[1, 2], [2, 3], [0, 1]]))
        a = Ad_array(np.array([1, 2, 3]), J)
        b = 3**a
        bJac = np.array(
            [
                [3 * np.log(3) * 1, 3 * np.log(3) * 2],
                [9 * np.log(3) * 2, 9 * np.log(3) * 3],
                [27 * np.log(3) * 0, 27 * np.log(3) * 1],
            ]
        )

        self.assertTrue(np.all(b.val == [3, 9, 27]))
        self.assertTrue(np.all(b.jac.A == bJac))

    def test_div_advar_scalar(self):
        a = Ad_array(10, 6)
        b = 2
        c = a / b
        self.assertTrue(c.val == 5, c.jac == 2)

    def test_div_advar_advar(self):
        # a = x ^ 3: b = x^2: x = 2
        a = Ad_array(8, 12)
        b = Ad_array(4, 4)
        c = a / b
        self.assertTrue(c.val == 2 and np.allclose(c.jac, 1))

    def test_full_jac(self):
        J = np.array(
            [
                [1, 3, 5, 1, 2],
                [1, 5, 1, 2, 5],
                [6, 2, 4, 6, 0],
                [2, 4, 1, 9, 9],
                [6, 2, 1, 45, 2],
            ]
        )

        a = Ad_array(np.array([1, 2, 3, 4, 5]), J.copy())  # np.array([J1, J2]))

        self.assertTrue(np.sum(a.full_jac() != J) == 0)

    def test_copy_scalar(self):
        a = Ad_array(1, 0)
        b = a.copy()
        self.assertTrue(a.val == b.val)
        self.assertTrue(a.jac == b.jac)
        a.val = 2
        a.jac = 3
        self.assertTrue(b.val == 1)
        self.assertTrue(b.jac == 0)

    def test_copy_vector(self):
        a = Ad_array(np.ones((3, 1)), np.ones((3, 1)))
        b = a.copy()
        self.assertTrue(np.allclose(a.val, b.val))
        self.assertTrue(np.allclose(a.jac, b.jac))
        a.val[0] = 3
        a.jac[2] = 4
        self.assertTrue(np.allclose(b.val, np.ones((3, 1))))
        self.assertTrue(np.allclose(b.jac, np.ones((3, 1))))


if __name__ == "__main__":
    unittest.main()
